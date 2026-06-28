# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import flash_attention

__all__ = ['WanModel']

T5_CONTEXT_TOKEN_NUMBER = 512
FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER = 257 * 2


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs, frame_offset=0):
    n, c = x.size(2), x.size(3) // 2

    # split freqs: temporal, height, width
    freqs = freqs.split(
        [c - 2 * (c // 3), c // 3, c // 3],
        dim=1
    )

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # --------------------------------------------------
        # 1. 当前样本 token 转 complex
        # --------------------------------------------------
        x_i = torch.view_as_complex(
            x[i, :seq_len]
            .to(torch.float64)
            .reshape(seq_len, n, -1, 2)
        )

        # --------------------------------------------------
        # 2. 只偏移时间维 RoPE
        # --------------------------------------------------
        t_start = int(frame_offset)
        t_end = t_start + int(f)

        if t_end > freqs[0].shape[0]:
            raise ValueError(
                f"RoPE temporal index out of range: "
                f"frame_offset={t_start}, f={f}, "
                f"required={t_end}, but freqs[0] length={freqs[0].shape[0]}"
            )

        freqs_t = freqs[0][t_start:t_end]
        freqs_h = freqs[1][:h]
        freqs_w = freqs[2][:w]

        # --------------------------------------------------
        # 3. 构造 F/H/W 三维 RoPE multiplier
        # --------------------------------------------------
        freqs_i = torch.cat(
            [
                freqs_t
                .view(f, 1, 1, -1)
                .expand(f, h, w, -1),

                freqs_h
                .view(1, h, 1, -1)
                .expand(f, h, w, -1),

                freqs_w
                .view(1, 1, w, -1)
                .expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)

        # --------------------------------------------------
        # 4. apply rotary embedding
        # --------------------------------------------------
        x_i = torch.view_as_real(
            x_i * freqs_i
        ).flatten(2)

        # 保留 padding 部分
        x_i = torch.cat(
            [x_i, x[i, seq_len:]],
            dim=0,
        )

        output.append(x_i)

    return torch.stack(output).float()


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        eps=1e-6,
    ):
        assert dim % num_heads == 0
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        self.norm_q = (
            WanRMSNorm(dim, eps=eps)
            if qk_norm
            else nn.Identity()
        )
        self.norm_k = (
            WanRMSNorm(dim, eps=eps)
            if qk_norm
            else nn.Identity()
        )

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        frame_offset=0,
        prev_kv_cache=None,
        sink_kv_cache=None,
        return_kv_cache=False,
        return_head_kv_cache=False,
        cache_frames=0,
        cache_query_frames=0,
        cache_strength=0.0,
    ):
        r"""
        Args:
            x:
                Tensor，形状 [B, L, C]

            seq_lens:
                Tensor，形状 [B]，每个样本的有效 token 长度。

            grid_sizes:
                Tensor，形状 [B, 3]，每行为 (F, H, W)。

            freqs:
                RoPE 频率。

            frame_offset:
                当前分段在全局 latent 时间轴上的起始位置。

                第一段通常为 0；
                第二段通常为：
                    latent_frames - latent_overlap

            prev_kv_cache:
                前一段相同去噪步、相同 Transformer block 的 KV：
                {
                    "k": [B, L_cache, num_heads, head_dim],
                    "v": [B, L_cache, num_heads, head_dim],
                    "lens": [B]
                }

                其中 k 已经应用过前一段对应的 RoPE。

            return_kv_cache:
                是否返回当前段尾部 KV。

            cache_frames:
                保存当前段最后多少个 latent 帧的 KV。

            cache_query_frames:
                当前段最前多少个 latent 帧可以读取前段 KV。

            cache_strength:
                KV cache 残差分支的强度。
        """
        b, s = x.shape[:2]
        n = self.num_heads
        d = self.head_dim

        # 1. 当前段 Q、K、V
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        # 2. 应用带全局时间偏移的 RoPE
        q_rope = rope_apply(
            q,
            grid_sizes,
            freqs,
            frame_offset=frame_offset,
        )

        k_rope = rope_apply(
            k,
            grid_sizes,
            freqs,
            frame_offset=frame_offset,
        )

        
        # 3. 当前段原始 self-attention
        current_out = flash_attention(
            q=q_rope,
            k=k_rope,
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size,
        )

        out = current_out

        # 4. 前段 KV cache 残差注意力
        use_prev_cache = (
            (prev_kv_cache is not None or sink_kv_cache is not None)
            and cache_strength > 0.0
            and cache_query_frames > 0
        )
        if return_kv_cache or prev_kv_cache is not None or cache_strength > 0.0:
            print(
                "[WanSelfAttention KV Gate]",
                "prev_kv_cache is None =", prev_kv_cache is None,
                "prev_keys =",
                (list(prev_kv_cache.keys())if prev_kv_cache is not None else None),
                "return_kv_cache =", return_kv_cache,
                "cache_frames =", cache_frames,
                "cache_query_frames =", cache_query_frames,
                "cache_strength =", cache_strength,
                "use_prev_cache =", use_prev_cache,
            )
        #读取上一段保存下来的历史 K/V cache
        if use_prev_cache:
            cache_k_list = []
            cache_v_list = []
            cache_lens_list = []

            if sink_kv_cache is not None:
                sink_k = sink_kv_cache["k"].to(device=q_rope.device,dtype=q_rope.dtype,)
                sink_v = sink_kv_cache["v"].to(device=v.device,dtype=v.dtype,    )
                sink_lens = sink_kv_cache["lens"].to(device=seq_lens.device,dtype=seq_lens.dtype,)

                cache_k_list.append(sink_k)
                cache_v_list.append(sink_v)
                cache_lens_list.append(sink_lens)

            if prev_kv_cache is not None:
                tail_k = prev_kv_cache["k"].to(device=q_rope.device,dtype=q_rope.dtype,)
                tail_v = prev_kv_cache["v"].to(device=v.device,dtype=v.dtype,)
                tail_lens = prev_kv_cache["lens"].to(device=seq_lens.device,dtype=seq_lens.dtype,)

                cache_k_list.append(tail_k)
                cache_v_list.append(tail_v)
                cache_lens_list.append(tail_lens)

            cached_k = torch.cat(cache_k_list, dim=1)
            cached_v = torch.cat(cache_v_list, dim=1)
            cached_lens = sum(cache_lens_list)

            if not torch.all(grid_sizes == grid_sizes[0]):
                raise ValueError(
                    "当前实现要求同一 batch 内所有样本具有相同的 "
                    "(F, H, W)。"
                )

            f, h, w = [
                int(value)
                for value in grid_sizes[0].tolist()
            ]

            tokens_per_frame = h * w

            # 只让当前段最前面的若干 latent 帧读取前段 cache。
            query_token_num = min(
                cache_query_frames * tokens_per_frame,
                f * tokens_per_frame,
                s,
            )

            if query_token_num > 0:
                #当前段前几帧的q跟上一段后几帧的kv以及第一段最开始几帧的kv做attention
                q_head = q_rope[:, :query_token_num]
                cache_out_head = flash_attention(
                    q=q_head,
                    k=cached_k,
                    v=cached_v,
                    k_lens=cached_lens, 
                    # cache 中只有前段尾部token，
                    # 这里不再使用原本的局部窗口限制。
                    window_size=(-1, -1),
                )
                # ==================================================
                # Debug: 确认上一段 KV cache 真的被当前段 Q 读取
                # ==================================================
                if not hasattr(self, "_kv_active_debug_printed"):
                    self._kv_active_debug_printed = True

                    current_norm = current_out[:, :query_token_num].float().norm().item()
                    cache_norm = cache_out_head.float().norm().item()

                    print(
                        "[WanSelfAttention KV ACTIVE]",
                        "cache_strength =", cache_strength,
                        "query_token_num =", query_token_num,
                        "q_head_shape =", tuple(q_head.shape),
                        "cached_k_shape =", tuple(cached_k.shape),
                        "cached_v_shape =", tuple(cached_v.shape),
                        "cached_lens =", cached_lens.detach().cpu().tolist(),
                        "current_norm =", current_norm,
                        "cache_norm =", cache_norm,
                        "ratio =", cache_norm / (current_norm + 1e-6),
                    )

                # 只修改当前段头部 query 对应的输出。
                cache_out = torch.zeros_like(current_out)
                cache_out[:, :query_token_num] = cache_out_head

                # 保留当前段自己的 attention，前段 cache 只作为残差补充。
                out = current_out + cache_strength * cache_out

        
        # 5. 保存当前段尾部 KV
        new_kv_cache = None
        if return_kv_cache and cache_frames > 0:
            cached_k_list = []
            cached_v_list = []
            cached_lens_list = []
            
            cached_head_k_list = []
            cached_head_v_list = []
            cached_head_lens_list = []

            for batch_idx in range(b):
                f, h, w = [
                    int(value)
                    for value
                    in grid_sizes[batch_idx].tolist()
                ]

                tokens_per_frame = h * w
                video_token_num = f * tokens_per_frame

                # seq_lens 可能包含 padding 后的有效长度信息。
                valid_len = min(
                    int(seq_lens[batch_idx].item()),
                    video_token_num,
                    s,
                )

                cache_token_num = min(
                    cache_frames * tokens_per_frame,
                    valid_len,
                )

                start_idx = valid_len - cache_token_num
                end_idx = valid_len

                # 缓存已经应用过正确全局时间位置的 K。
                cached_k_list.append(
                    k_rope[
                        batch_idx,
                        start_idx:end_idx,
                    ].detach().to(device="cpu",dtype=torch.float16,).clone()
                )

                cached_v_list.append(
                    v[
                        batch_idx,
                        start_idx:end_idx,
                    ].detach().to(device="cpu",dtype=torch.float16,).clone())

                cached_lens_list.append(cache_token_num)
                
                if return_head_kv_cache:
                    head_start_idx = 0
                    head_end_idx = cache_token_num

                    cached_head_k_list.append(k_rope[batch_idx,head_start_idx:head_end_idx,].detach().to(device="cpu", dtype=torch.float16).clone())

                    cached_head_v_list.append(v[batch_idx,head_start_idx:head_end_idx,].detach().to(device="cpu", dtype=torch.float16).clone())

                    cached_head_lens_list.append(cache_token_num)

            # 当前项目一般 batch 内视频尺寸一致。
            # 若不同，则需要 padding cache。
            if len(set(cached_lens_list)) != 1:
                raise ValueError(
                    "当前 batch 内不同样本的 KV cache 长度不同，"
                    "需要为 cache 增加 padding。"
                )

            new_kv_cache = {
                "k": torch.stack(
                    cached_k_list,
                    dim=0,
                ),
                "v": torch.stack(
                    cached_v_list,
                    dim=0,
                ),
                "lens": torch.tensor(
                    cached_lens_list,
                    device="cpu",
                    dtype=seq_lens.dtype,
                ),
            }
            if return_head_kv_cache:
                new_kv_cache.update({
                    "head_k": torch.stack(cached_head_k_list, dim=0),
                    "head_v": torch.stack(cached_head_v_list, dim=0),
                    "head_lens": torch.tensor(
                        cached_head_lens_list,
                        dtype=seq_lens.dtype,
                    ),
                })
        # 6. 输出映射
        out = out.flatten(2)
        out = self.o(out)

        if return_kv_cache:
            return out, new_kv_cache

        return out


class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        image_context_length = context.shape[1] - T5_CONTEXT_TOKEN_NUMBER
        context_img = context[:, :image_context_length]
        context = context[:, image_context_length:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
}


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
    def _get_hidden_fusion_runtime(self):
        runtime = getattr(self, "hidden_fusion_runtime", None)
        block_idx = getattr(self, "hidden_fusion_block_idx", -1)
        return runtime, block_idx

    def _hidden_tail_head_fusion_and_cache(self, y, grid_sizes):
        """
        对 self-attention 输出 y 做跨段 hidden state 融合，并保存当前段尾部 hidden state。

        y: [B, L, C]
        grid_sizes: [B, 3], 每一行是 (F, H, W)
        """
        runtime, block_idx = self._get_hidden_fusion_runtime()

        if runtime is None:
            return y

        block_start = runtime.get("block_start", 0)
        block_end = runtime.get("block_end", 10**9)
        # debug: 只打印 block_start 这一层，避免刷屏
        debug = runtime.get("debug", False)
        if debug and block_idx == block_start:
            print(
                f"[HF-Block-Enter] block={block_idx}, "
                f"enabled={runtime.get('enabled', False)}, "
                f"return_cache={runtime.get('return_cache', False)}, "
                f"prev_cache_none={runtime.get('prev_cache', None) is None}, "
                f"cur_cache_none={runtime.get('cur_cache', None) is None}, "
                f"alpha={runtime.get('alpha', None)}, "
                f"overlap={runtime.get('overlap', None)}, "
                f"block_range=({block_start},{block_end})"
            )
        # 只在指定 block 范围内做
        if not (block_start <= block_idx < block_end):
            return y

        # 当前先只支持单视频 B=1，和你的生成流程一致
        if y.shape[0] != 1:
            return y

        f, h, w = grid_sizes[0].tolist()
        f = int(f)
        h = int(h)
        w = int(w)

        S = h * w
        overlap = int(runtime.get("overlap", 2))
        current_overlap = min(overlap, f)

        hidden_tokens = current_overlap * S
        if debug and block_idx == block_start:
            print(
                f"[HF-Shape] block={block_idx}, "
                f"y={tuple(y.shape)}, "
                f"grid=(f={f}, h={h}, w={w}), "
                f"S={S}, current_overlap={current_overlap}, "
                f"hidden_tokens={hidden_tokens}"
            )

        if hidden_tokens <= 0 or hidden_tokens > y.shape[1]:
            return y

        # ============================================================
        # 1. 用上一段 hidden tail 融合当前段 hidden head
        # ============================================================
        enabled = runtime.get("enabled", False)
        prev_cache = runtime.get("prev_cache", None)
        alpha = float(runtime.get("alpha", 0.03))

        if enabled and prev_cache is not None and alpha > 0:
            prev_tail = None

            if isinstance(prev_cache, dict):
                prev_tail = prev_cache.get(block_idx, None)
            elif isinstance(prev_cache, (list, tuple)):
                if block_idx < len(prev_cache):
                    prev_tail = prev_cache[block_idx]

            if prev_tail is not None:
                prev_tail = prev_tail.to(device=y.device, dtype=y.dtype)

                if prev_tail.ndim == 3:
                    use_tokens = min(hidden_tokens, prev_tail.shape[1], y.shape[1])

                    if use_tokens > 0:
                        # 避免原地修改导致 autograd / amp 下潜在问题
                        y = y.clone()
                        y[:, :use_tokens, :] = (
                            (1.0 - alpha) * y[:, :use_tokens, :]
                            + alpha * prev_tail[:, -use_tokens:, :]
                        )

        # ============================================================
        # 2. 保存当前段尾部 hidden state，供下一段使用
        # ============================================================
        if runtime.get("return_cache", False):
            cur_cache = runtime.get("cur_cache", None)
            if cur_cache is not None:
                cur_cache[block_idx] = (
                    y[:, -hidden_tokens:, :].detach().cpu().clone()
                )

        return y
    def forward(
    self,
    x,
    e,
    seq_lens,
    grid_sizes,
    freqs,
    context,
    context_lens,
    frame_offset=0,
    prev_kv_cache=None,
    sink_kv_cache=None,
    return_kv_cache=False,
    return_head_kv_cache=False,
    cache_frames=0,
    cache_query_frames=0,
    cache_strength=0.0,
    enable_hidden_state_fusion=True,
):
        r"""
    Args:
        x(Tensor):
            Shape [B, L, C]

        e(Tensor):
            Shape [B, 6, C]

        seq_lens(Tensor):
            Shape [B], length of each sequence in batch

        grid_sizes(Tensor):
            Shape [B, 3], the second dimension contains (F, H, W)

        freqs(Tensor):
            RoPE freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32

        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e).chunk(6, dim=1)

        assert e[0].dtype == torch.float32

    
    # 1. self-attention with optional KV cache
        attn_input = self.norm1(x).float() * (1 + e[1]) + e[0]
        if return_kv_cache or prev_kv_cache is not None:
            print(
                "[WanAttentionBlock before self_attn]",
                "block_idx =", getattr(self, "hidden_fusion_block_idx", None),
                "prev_kv_cache is None =", prev_kv_cache is None,
                "prev_keys =",
                (list(prev_kv_cache.keys())if prev_kv_cache is not None else None),
                "return_kv_cache =", return_kv_cache,
                "cache_frames =", cache_frames,
                "cache_query_frames =", cache_query_frames,
                "cache_strength =", cache_strength,
            )
        # 把当前 block 的输入送进 self-attention
        attn_result = self.self_attn(
            attn_input,
            seq_lens,
            grid_sizes,
            freqs,
            frame_offset=frame_offset,
            prev_kv_cache=prev_kv_cache,
            sink_kv_cache=sink_kv_cache,
            return_kv_cache=return_kv_cache,
            return_head_kv_cache=return_head_kv_cache,
            cache_frames=cache_frames,
            cache_query_frames=cache_query_frames,
            cache_strength=cache_strength,
        )

        if return_kv_cache:
            y, new_kv_cache = attn_result
        else:
            y = attn_result
            new_kv_cache = None

    
    # 2. hidden state fusion
        if enable_hidden_state_fusion:
            y = self._hidden_tail_head_fusion_and_cache(y, grid_sizes)

        with amp.autocast(dtype=torch.float32):
            x = x + y * e[2]
 
   
    # 3. cross-attention & FFN
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(
            self.norm3(x),
            context,
            context_lens,
        )

            y = self.ffn(
            self.norm2(x).float() * (1 + e[4]) + e[3]
        )

            with amp.autocast(dtype=torch.float32):
                x = x + y * e[5]

            return x

        x = cross_attn_ffn(
        x,
        context,
        context_lens,
        e,
    )

        if return_kv_cache:
            return x, new_kv_cache

        return x    


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim, flf_pos_emb=False):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))
        if flf_pos_emb:  # NOTE: we only use this for `flf2v`
            self.emb_pos = nn.Parameter(
                torch.zeros(1, FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER, 1280))

    def forward(self, image_embeds):
        if hasattr(self, 'emb_pos'):
            bs, n, d = image_embeds.shape
            image_embeds = image_embeds.view(-1, 2 * n, d)
            image_embeds = image_embeds + self.emb_pos
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video) or 'flf2v' (first-last-frame-to-video) or 'vace'
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'flf2v', 'vace']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)

        if model_type == 'i2v' or model_type == 'flf2v':
            self.img_emb = MLPProj(1280, dim, flf_pos_emb=model_type == 'flf2v')

        # initialize weights
        self.init_weights()

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,

        # KV cache / RoPE offset 新增参数
        frame_offset=0,
        prev_kv_cache=None,
        sink_kv_cache=None,
        return_kv_cache=False,
        return_head_kv_cache=False,
        cache_frames=0,
        cache_query_frames=0,
        cache_strength=0.0,
        enable_hidden_state_fusion=True,
        kv_block_start=8,
        kv_block_end=20,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]

            t (Tensor):
                Diffusion timesteps tensor of shape [B]

            context (List[Tensor]):
                List of text embeddings each with shape [L, C]

            seq_len (int):
                Maximum sequence length for positional encoding

            clip_fea (Tensor, optional):
                CLIP image features for image-to-video mode or first-last-frame-to-video mode

            y (List[Tensor], optional):
                Conditional video inputs for image-to-video mode, same shape as x

            frame_offset:
                当前分段在全局 latent 时间轴上的起始 index。

            prev_kv_cache:
                前一段同一 diffusion step 的 KV cache。
                结构：
                {
                    block_idx: {
                        "k": Tensor,
                        "v": Tensor,
                        "lens": Tensor
                    }
                }
            kv_block_start / kv_block_end:
                只在 [kv_block_start, kv_block_end) 范围内的 block 使用 KV cache。
                
            sink_kv_cache:
                第一段开头几帧保存下来的长期锚点 KV cache。
        """

        if self.model_type == 'i2v' or self.model_type == 'flf2v':
            assert clip_fea is not None and y is not None

       
        # 1. params
        device = self.patch_embedding.weight.device

        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [
                torch.cat([u, v], dim=0)
                for u, v in zip(x, y)
            ]

        
        # 2. patch embedding
        x = [
            self.patch_embedding(u.unsqueeze(0))
            for u in x
        ]

        grid_sizes = torch.stack(
            [
                torch.tensor(
                    u.shape[2:],
                    dtype=torch.long,
                    device=device,
                )
                for u in x
            ]
        )

        x = [
            u.flatten(2).transpose(1, 2)
            for u in x
        ]

        seq_lens = torch.tensor(
            [u.size(1) for u in x],
            dtype=torch.long,
            device=device,
        )

        assert seq_lens.max() <= seq_len

        x = torch.cat(
            [
                torch.cat(
                    [
                        u,
                        u.new_zeros(
                            1,
                            seq_len - u.size(1),
                            u.size(2),
                        ),
                    ],
                    dim=1,
                )
                for u in x
            ]
        )

        
        # 3. time embeddings
        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(
                    self.freq_dim,
                    t,
                ).float()
            )

            e0 = self.time_projection(e).unflatten(
                1,
                (6, self.dim),
            )

            assert e.dtype == torch.float32
            assert e0.dtype == torch.float32

        
        # 4. text / image context
        context_lens = None

        context = self.text_embedding(
            torch.stack(
                [
                    torch.cat(
                        [
                            u,
                            u.new_zeros(
                                self.text_len - u.size(0),
                                u.size(1),
                            ),
                        ]
                    )
                    for u in context
                ]
            )
        )

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)
            context = torch.concat(
                [context_clip, context],
                dim=1,
            )

        
        # 5. transformer blocks
        hidden_fusion_runtime = getattr(
            self,
            "hidden_fusion_runtime",
            None,
        )

        new_kv_cache = {} if return_kv_cache else None

        for block_idx, block in enumerate(self.blocks):
            block.hidden_fusion_runtime = hidden_fusion_runtime
            block.hidden_fusion_block_idx = block_idx

            # 只在指定 block 范围内启用 KV cache
            block_use_kv = (
                block_idx >= kv_block_start
                and block_idx < kv_block_end
            )

            # 取出前一段中对应 block 的 cache和 sink cache
            block_prev_kv_cache = None
            block_sink_kv_cache = None
            if block_use_kv and prev_kv_cache is not None:
                block_prev_kv_cache = prev_kv_cache.get(
                    block_idx,
                    None,
                )
            if block_use_kv and sink_kv_cache is not None:
                block_sink_kv_cache = sink_kv_cache.get(block_idx, None)
            if return_kv_cache and block_idx in [12, 13, 14, 15]:
                print(
                    "[WanModel KV Dispatch]",
                    "block_idx =", block_idx,
                    "block_use_kv =", block_use_kv,
                    "prev_kv_cache is None =", prev_kv_cache is None,
                    "block_prev_kv_cache is None =", block_prev_kv_cache is None,
                    "block_prev_keys =",
                    (list(block_prev_kv_cache.keys()) if block_prev_kv_cache is not None else None),
                    "return_kv_cache =", return_kv_cache and block_use_kv,
                    "cache_frames =", cache_frames if block_use_kv else 0,
                    "cache_query_frames =", cache_query_frames if block_use_kv else 0,
                    "cache_strength =", cache_strength if block_use_kv else 0.0,
                )
            block_result = block(
                x,
                e=e0,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                freqs=self.freqs,
                context=context,
                context_lens=context_lens,

                frame_offset=frame_offset,
                prev_kv_cache=block_prev_kv_cache,
                sink_kv_cache=block_sink_kv_cache,
                # 只有指定 block 保存 / 读取 KV
                return_kv_cache=(
                    return_kv_cache and block_use_kv
                ),
                return_head_kv_cache=(
                    return_head_kv_cache and return_kv_cache and block_use_kv
                ),
                cache_frames=(
                    cache_frames if block_use_kv else 0
                ),
                cache_query_frames=(
                    cache_query_frames if block_use_kv else 0
                ),
                cache_strength=(
                    cache_strength if block_use_kv else 0.0
                ),
                enable_hidden_state_fusion=enable_hidden_state_fusion,
            )

            if return_kv_cache and block_use_kv:
                x, block_new_kv_cache = block_result

                if block_new_kv_cache is not None:
                    new_kv_cache[block_idx] = block_new_kv_cache
            else:
                x = block_result


        # 6. head
        x = self.head(x, e)

        # 7. unpatchify
        x = self.unpatchify(
            x,
            grid_sizes,
        )

        output = [
            u.float()
            for u in x
        ]

        if return_kv_cache:
            return output, new_kv_cache

        return output

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
