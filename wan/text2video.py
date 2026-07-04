# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


def _make_latent_windows(total_latent_frames, window_size, stride):
    """Return inclusive-exclusive temporal windows over latent T."""
    if window_size <= 0:
        raise ValueError("latent_window_size must be positive.")
    if stride <= 0:
        raise ValueError("latent_stride must be positive.")
    if window_size > total_latent_frames:
        window_size = total_latent_frames

    windows = []
    start = 0
    while start + window_size <= total_latent_frames:
        windows.append((start, start + window_size))
        start += stride

    if not windows:
        windows.append((0, total_latent_frames))
    elif windows[-1][1] < total_latent_frames:
        windows.append((total_latent_frames - window_size, total_latent_frames))

    # Remove duplicate tail window when stride lands exactly at the end.
    deduped = []
    for item in windows:
        if item not in deduped:
            deduped.append(item)
    return deduped


def _window_blend_weight(length, mode, device, dtype):
    """Create per-frame weights for a latent window."""
    if mode == "flat" or length <= 2:
        return torch.ones(1, length, 1, 1, device=device, dtype=dtype)
    if mode != "triangle":
        raise ValueError(f"Unsupported window_weight: {mode}")

    pos = torch.linspace(0, 1, length, device=device, dtype=dtype)
    # Center frames get weight 1, edge frames get weight 0.5.
    weight = 0.5 + 0.5 * (1.0 - (2.0 * pos - 1.0).abs())
    return weight.view(1, length, 1, 1)


def _lerp_context(contexts, progress, enable_lerp=True):
    """
    Interpolate between prompt contexts.

    contexts is the list returned by Wan T5 encoder, one tensor per prompt.
    If neighboring prompt tensors have different shapes, fall back to nearest
    prompt selection because torch.lerp requires equal shapes.
    """
    if len(contexts) == 1:
        return [contexts[0]]

    progress = max(0.0, min(1.0, float(progress)))
    scaled = progress * (len(contexts) - 1)
    left = int(scaled)
    right = min(left + 1, len(contexts) - 1)
    alpha = scaled - left

    if (not enable_lerp) or left == right:
        return [contexts[left if alpha < 0.5 else right]]

    a, b = contexts[left], contexts[right]
    if a.shape != b.shape:
        return [a if alpha < 0.5 else b]
    return [torch.lerp(a, b, alpha)]

class WanT2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False)

        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (
                usp_attn_forward,
                usp_dit_forward,
            )
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt

    def generate(self,
                 input_prompt,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 # 多段生成使用自己的的提示词
                 context_override=None,
                 context_null_override=None,
                 # kv cache
                 return_kv_cache=False,
                 return_head_kv_cache=False,
                 prev_kv_cache_cond=None,
                 sink_kv_cache_cond=None,
                 kv_cache_frames=2,
                 kv_cache_query_frames=2,
                 kv_cache_strength=0.08,
                 kv_cache_use_start_ratio=0.45,
                 kv_cache_use_end_ratio=0.75,
                 kv_block_start=12,
                 kv_block_end=16,
                 latent_frame_offset=0,
                 # text cross-attention cache, text_out=attn_map@V_text
                 prev_text_cross_cache_cond=None,
                 return_text_cross_cache=False,
                 text_cross_cache_frames=2,
                 text_cross_query_frames=1,
                 text_cross_strength=0.03,
text_cross_use_start_ratio=0.20,
text_cross_use_end_ratio=0.50,
text_cross_block_start=10,
text_cross_block_end=16,
prev_initial_state_cache_cond=None,
return_initial_state_cache=False,
initial_step_start=0,
initial_step_end=2,
initial_strength=0.1,
):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (tupele[`int`], *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        F = frame_num
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if context_override is not None:
            context = context_override
        else:
            if not self.t5_cpu:
                self.text_encoder.model.to(self.device)
                context = self.text_encoder([input_prompt], self.device)
                if offload_model:
                    self.text_encoder.model.cpu()
            else:
                context = self.text_encoder([input_prompt], torch.device('cpu'))
                context = [t.to(self.device) for t in context]

        if context_null_override is not None:
            context_null = context_null_override
        else:
            if not self.t5_cpu:
                self.text_encoder.model.to(self.device)
                context_null = self.text_encoder([n_prompt], self.device)
                if offload_model:
                    self.text_encoder.model.cpu()
            else:
                context_null = self.text_encoder([n_prompt], torch.device('cpu'))
                context_null = [t.to(self.device) for t in context_null]

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noise

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}
            kv_cache_cond = [] if return_kv_cache else None
            text_cross_cache_cond = [] if return_text_cross_cache else None
            initial_state_cache_cond = [] if return_initial_state_cache else None
            num_steps = len(timesteps)

            for step_idx, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = [t]
                timestep = torch.stack(timestep)

                self.model.to(self.device)
                # 在特定去噪步骤范围内使用kv缓存
                kv_cache_start_step = int(num_steps * kv_cache_use_start_ratio)
                kv_cache_end_step = int(num_steps * kv_cache_use_end_ratio)

                use_kv_cache = (
                    prev_kv_cache_cond is not None
                    and kv_cache_start_step <= step_idx < kv_cache_end_step
                    and step_idx < len(prev_kv_cache_cond)
                    and prev_kv_cache_cond[step_idx] is not None
                )   
                prev_step_kv_cond = (prev_kv_cache_cond[step_idx]if use_kv_cache else None)
                
                use_sink_kv_cache = (
                    sink_kv_cache_cond is not None
                    and kv_cache_start_step <= step_idx < kv_cache_end_step
                    and step_idx < len(sink_kv_cache_cond)
                    and sink_kv_cache_cond[step_idx] is not None
                )
                sink_step_kv_cond = (sink_kv_cache_cond[step_idx]if use_sink_kv_cache else None)

                use_any_kv_cache = use_kv_cache or use_sink_kv_cache
               
                cur_kv_cond = None
                should_return_kv_cache = (return_kv_cache and kv_cache_start_step <= step_idx < kv_cache_end_step)
                # 在特定去噪步骤范围内使用文本交叉注意力缓存
                text_cross_start_step = int(num_steps * text_cross_use_start_ratio)
                text_cross_end_step = int(num_steps * text_cross_use_end_ratio)

                use_prev_text_cross_cache = (
                    prev_text_cross_cache_cond is not None
                    and text_cross_start_step <= step_idx < text_cross_end_step
                    and step_idx < len(prev_text_cross_cache_cond)
                    and prev_text_cross_cache_cond[step_idx] is not None
                )

                prev_step_text_cross_cache = (
                    prev_text_cross_cache_cond[step_idx]
                    if use_prev_text_cross_cache else None
                )

                should_return_text_cross_cache = (
                    return_text_cross_cache
                    and text_cross_start_step <= step_idx < text_cross_end_step
                )
                # 在特定去噪步骤范围内使用初始状态缓存
                use_prev_initial_state_cache = (
                    prev_initial_state_cache_cond is not None
                    and initial_step_start <= step_idx < initial_step_end
                    and step_idx < len(prev_initial_state_cache_cond)
                    and prev_initial_state_cache_cond[step_idx] is not None
                )
                prev_step_initial_state_cache = (
                    prev_initial_state_cache_cond[step_idx]
                    if use_prev_initial_state_cache else None
                )
                should_return_initial_state_cache = (
                    return_initial_state_cache
                    and initial_step_start <= step_idx < initial_step_end
                )
                cond_out = self.model(
                    latent_model_input,
                    t=timestep,
                    **arg_c,
                    # KV cache
                    frame_offset=latent_frame_offset,
                    prev_kv_cache=prev_step_kv_cond,
                    sink_kv_cache=sink_step_kv_cond,
                    return_kv_cache=should_return_kv_cache,
                    return_head_kv_cache=(
        return_head_kv_cache and should_return_kv_cache
    ),
    cache_frames=kv_cache_frames,
    cache_query_frames=(
        kv_cache_query_frames if use_kv_cache else 0
    ),
    cache_strength=(
        kv_cache_strength if use_kv_cache else 0.0
    ),
    kv_block_start=kv_block_start,
    kv_block_end=kv_block_end,
                    prev_text_cross_cache=prev_step_text_cross_cache,
                    return_text_cross_cache=should_return_text_cross_cache,
                    text_cross_cache_frames=text_cross_cache_frames,
                    text_cross_query_frames=text_cross_query_frames,
                    text_cross_strength=text_cross_strength,
                    text_cross_block_start=text_cross_block_start,
                    text_cross_block_end=text_cross_block_end,
                    initial_strength=initial_strength,
                    prev_initial_state_cache=prev_step_initial_state_cache,
                    return_initial_state_cache=should_return_initial_state_cache,
                    
                )

                
                noise_pred_cond_list, cur_kv_cond, cur_text_cross_cache, cur_initial_state_cache = cond_out
                noise_pred_cond = noise_pred_cond_list[0]

                if return_kv_cache:
                    kv_cache_cond.append(cur_kv_cond)
                if return_text_cross_cache:
                    text_cross_cache_cond.append(cur_text_cross_cache)
                if return_initial_state_cache:
                    initial_state_cache_cond.append(cur_initial_state_cache)

                noise_pred_uncond = self.model(
                    latent_model_input,
                    t=timestep,
                    **arg_null,
                    frame_offset=latent_frame_offset,
                    prev_text_cross_cache=None,
                    return_text_cross_cache=False,
                    text_cross_cache_frames=text_cross_cache_frames,
                    text_cross_query_frames=text_cross_query_frames,
                    text_cross_strength=0.0,
                    text_cross_block_start=text_cross_block_start,
                    text_cross_block_end=text_cross_block_end,
                )[0][0]                
                
                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]

            x0 = latents
            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()
        video = videos[0] if self.rank == 0 else None
        return kv_cache_cond,text_cross_cache_cond,initial_state_cache_cond, video
        
    def generate_sliding_window(
        self,
        input_prompts,
        size=(1280, 720),
        frame_num=81,
        shift=5.0,
        sample_solver="unipc",
        sampling_steps=50,
        guide_scale=5.0,
        n_prompt="",
        seed=-1,
        offload_model=True,
        latent_window_size=13,
        latent_stride=4,
        window_weight="triangle",
        prompt_lerp=True,
    ):
        """
        Generate long T2V with temporal latent windows and velocity blending.
        """
        if isinstance(input_prompts, str):
            input_prompts = [input_prompts]
        input_prompts = [p.strip() for p in input_prompts if p and p.strip()]
        if not input_prompts:
            raise ValueError("input_prompts must contain at least one prompt.")

        F = frame_num
        target_shape = (
            self.vae.model.z_dim,
            (F - 1) // self.vae_stride[0] + 1,
            size[1] // self.vae_stride[1],
            size[0] // self.vae_stride[2],
        )
        total_latent_frames = target_shape[1]
        windows = _make_latent_windows(
            total_latent_frames,
            latent_window_size,
            latent_stride,
        )

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            contexts = self.text_encoder(input_prompts, self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            contexts = self.text_encoder(input_prompts, torch.device("cpu"))
            context_null = self.text_encoder([n_prompt], torch.device("cpu"))
            contexts = [t.to(self.device) for t in contexts]
            context_null = [t.to(self.device) for t in context_null]

        latent = torch.randn(
            target_shape[0],
            target_shape[1],
            target_shape[2],
            target_shape[3],
            dtype=torch.float32,
            device=self.device,
            generator=seed_g,
        )

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, "no_sync", noop_no_sync)

        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
            if sample_solver == "unipc":
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False,
                )
                sample_scheduler.set_timesteps(
                    sampling_steps,
                    device=self.device,
                    shift=shift,
                )
                timesteps = sample_scheduler.timesteps
            elif sample_solver == "dpm++":
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False,
                )
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas,
                )
            else:
                raise NotImplementedError("Unsupported solver.")

            self.model.to(self.device)

            logging.info(
                "[SlidingWindow] frame_num=%s latent_T=%s windows=%s "
                "latent_window_size=%s latent_stride=%s prompts=%s",
                frame_num,
                total_latent_frames,
                windows,
                latent_window_size,
                latent_stride,
                len(input_prompts),
            )

            for _, t in enumerate(tqdm(timesteps)):
                velocity_acc = torch.zeros_like(latent)
                weight_acc = torch.zeros_like(latent)
                timestep = torch.stack([t]).to(self.device)

                for start, end in windows:
                    latent_slice = latent[:, start:end, :, :]
                    window_t = end - start
                    seq_len = math.ceil(
                        (target_shape[2] * target_shape[3])
                        / (self.patch_size[1] * self.patch_size[2])
                        * window_t
                        / self.sp_size
                    ) * self.sp_size

                    center = (start + end - 1) * 0.5
                    progress = center / max(total_latent_frames - 1, 1)
                    context = _lerp_context(contexts, progress, prompt_lerp)

                    arg_c = {"context": context, "seq_len": seq_len}
                    arg_null = {"context": context_null, "seq_len": seq_len}
                    latent_model_input = [latent_slice]

                    velocity_cond = self.model(
                        latent_model_input,
                        t=timestep,
                        **arg_c,
                    )[0]
                    velocity_uncond = self.model(
                        latent_model_input,
                        t=timestep,
                        **arg_null,
                    )[0]
                    velocity = velocity_uncond + guide_scale * (
                        velocity_cond - velocity_uncond
                    )

                    weight = _window_blend_weight(
                        window_t,
                        window_weight,
                        latent.device,
                        latent.dtype,
                    )
                    velocity_acc[:, start:end, :, :] += velocity * weight
                    weight_acc[:, start:end, :, :] += weight

                    del latent_model_input, velocity_cond, velocity_uncond, velocity

                global_velocity = velocity_acc / weight_acc.clamp_min(1e-6)

                latent = sample_scheduler.step(
                    global_velocity.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g,
                )[0].squeeze(0)

                del velocity_acc, weight_acc, global_velocity, timestep
                if offload_model:
                    torch.cuda.empty_cache()

            x0 = [latent]
            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()

            if self.rank == 0:
                videos = self.vae.decode(x0)

        del latent
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None