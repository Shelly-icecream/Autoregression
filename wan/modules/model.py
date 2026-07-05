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
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
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

class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
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
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs,
                # kv cache
                frame_offset=0,
                prev_kv_cache=None,
        sink_kv_cache=None,
        return_kv_cache=False,
        return_head_kv_cache=False,
        cache_frames=0,
        cache_query_frames=0,
        cache_strength=0.0,):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)
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
            
            valid_len = int(seq_lens[0].item())

            current_q = q_rope[:, :valid_len]
            current_k = k_rope[:, :valid_len]
            current_v = v[:, :valid_len]
            
            all_k = torch.cat([current_k, cache_strength*cached_k], dim=1)
            all_v = torch.cat([current_v, cache_strength*cached_v], dim=1)
            all_lens = seq_lens + cached_lens

            out = flash_attention(
                q=current_q,
                k=all_k,
                v=all_v,
                k_lens=all_lens,
                window_size=(-1, -1),
            )
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
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
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

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        # kv cache
        frame_offset=0,
    prev_kv_cache=None,
    sink_kv_cache=None,
    return_kv_cache=False,
    return_head_kv_cache=False,
    cache_frames=0,
    cache_query_frames=0,
    cache_strength=0.0,
    enable_hidden_state_fusion=True,
        # text cross-attention cache
        prev_text_cross_cache=None,
        return_text_cross_cache=False,
        text_cross_cache_frames=2,
        text_cross_query_frames=1,
        text_cross_strength=0.03,
        initial_strength=0.1,
        initial_state=None,
        return_initial_state_cache=False,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32

        # self-attention with optional KV cache
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
        with amp.autocast(dtype=torch.float32):
            x = x + y * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            text_out=self.cross_attn(self.norm3(x), context, context_lens)
            x = x + text_out
            y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
            with amp.autocast(dtype=torch.float32):
                x = x + y * e[5]
                
            t = int(grid_sizes[0][0])
            h = int(grid_sizes[0][1])
            w = int(grid_sizes[0][2])
            tokens_per_frame = h * w
            
            if initial_state is not None:
                #x = x+initial_strength*initial_state.to(device=x.device, dtype=x.dtype)
                x= initial_state.to(device=x.device, dtype=x.dtype)
                
                
            if prev_text_cross_cache is not None and text_cross_strength > 0:
                prev_tail = prev_text_cross_cache.to(
                    device=x.device,
                    dtype=x.dtype,
                )
                query_token_num = min(
                    text_cross_query_frames * tokens_per_frame,
                    prev_tail.shape[1],
                    x.shape[1],
                )
                if query_token_num > 0:
                    prev_tail = prev_tail[:, -query_token_num:, :]
                    x[:, :query_token_num, :] = (
                        x[:, :query_token_num, :]
                        + text_cross_strength * prev_tail
                    )
                    
                    
            initial_state_cache=None
            if return_initial_state_cache:
                initial_state_cache = x.detach().clone()
            # 保存当前段尾部几帧的hidden state,这个是吸收了文本信息并且经过了MLP/FFN的hidden state
            text_cross_tail = None
            if return_text_cross_cache:
                tail_token_num = min(
                    text_cross_cache_frames * tokens_per_frame,
                    x.shape[1],
                )
                text_cross_tail = (
                    x[:, -tail_token_num:, :]
                    .detach()
                    .cpu()
                    .clone()
                )
            return x, text_cross_tail,initial_state_cache

        x, text_cross_tail, initial_state_cache = cross_attn_ffn(x, context, context_lens, e)
        return x,new_kv_cache,text_cross_tail,initial_state_cache


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
         # text cross-attention cache, text_out=attn_map@V_text
         prev_text_cross_cache=None,
         return_text_cross_cache=False,
         text_cross_cache_frames=2,
         text_cross_query_frames=1,
         text_cross_strength=0.03,
         text_cross_block_start=10,
         text_cross_block_end=16,
         initial_strength=0.1,
         prev_initial_state_cache=None,
         return_initial_state_cache=False,
         block_start_index=0,
         block_end_index=2,
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
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode or first-last-frame-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v' or self.model_type == 'flf2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 (x2) x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            )
        new_kv_cache = {} if return_kv_cache else None
        cur_text_cross_cache = {} if return_text_cross_cache else None
        cur_initial_state_cache = {} if return_initial_state_cache else None
        
        # 在特定的block范围内使用和保存text cross-attention cache
        # 在特定的block中使用和保存initial state
        for block_idx, block in enumerate(self.blocks):
            block_use_kv = (
                block_idx >= kv_block_start
                and block_idx < kv_block_end
            )
            use_text_cross_cache_block = (
                text_cross_block_start <= block_idx < text_cross_block_end
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

            # 取出前一段中对应 block 的 text cross-attention cache
            prev_block_text_cross = None
            if (
                prev_text_cross_cache is not None
                and use_text_cross_cache_block
                and block_idx in prev_text_cross_cache
            ):
                prev_block_text_cross = prev_text_cross_cache[block_idx]
                
            # 取出前一段中对应 block 的 initial state cache
            use_initial_state_block = (block_start_index <= block_idx < block_end_index)
            prev_block_initial_state = None
            if (
                prev_initial_state_cache is not None
                and use_initial_state_block
                and isinstance(prev_initial_state_cache, dict)
            ):
                prev_block_initial_state = prev_initial_state_cache.get(block_idx, None)

            x, block_new_kv_cache,block_text_cross_tail, block_initial_state_cache = block(
                x,
                **kwargs,
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
                prev_text_cross_cache=prev_block_text_cross,
                return_text_cross_cache=(
                    return_text_cross_cache and use_text_cross_cache_block
                ),
                text_cross_cache_frames=text_cross_cache_frames,
                text_cross_query_frames=text_cross_query_frames,
                text_cross_strength=text_cross_strength,
                initial_strength=initial_strength,
                initial_state=prev_block_initial_state,
                return_initial_state_cache=(return_initial_state_cache and use_initial_state_block)

            )

            if return_text_cross_cache and cur_text_cross_cache is not None and block_text_cross_tail is not None:
                cur_text_cross_cache[block_idx] = block_text_cross_tail
            if return_kv_cache and new_kv_cache is not None and block_new_kv_cache is not None:
                new_kv_cache[block_idx] = block_new_kv_cache
            if return_initial_state_cache and use_initial_state_block and block_initial_state_cache is not None:
                cur_initial_state_cache[block_idx] = block_initial_state_cache

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        out = [u.float() for u in x]


        return out,new_kv_cache, cur_text_cross_cache, cur_initial_state_cache

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
