# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import logging
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

import random

import torch
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_image, cache_video, str2bool



PROMPT_SEGMENTS = [
    {
        "name": "seg_01",
        "prompt": "A white jeep car is running on the beach, sunny.",
        "frame_num": 49,
        "seed": 42,
    },
    {
        "name": "seg_02",
        "prompt": "A white jeep car running on the beach, gradual transition from bright sunset to dark night sky, natural fading light.",
        "frame_num": 49,
        "seed": 42,
    },
]


NEGATIVE_PROMPT = (
    "low quality, blurry, distorted, deformed, bad anatomy, "
    "extra limbs, inconsistent character, flickering, text, watermark"
)

@torch.no_grad()
def encode_prompt_to_context(wan_t2v, prompt):
    """
    把一个 prompt 输入 Wan2.1 的 text_encoder，得到 context。

    Wan2.1 的 context 通常是 list[tensor]。
    每个 tensor 是文本条件嵌入，后面会传给 DiT model。
    """

    device = wan_t2v.device

    if not wan_t2v.t5_cpu:
        wan_t2v.text_encoder.model.to(device)
        context = wan_t2v.text_encoder([prompt], device)
        wan_t2v.text_encoder.model.cpu()
        torch.cuda.empty_cache()
    else:
        context = wan_t2v.text_encoder([prompt], torch.device("cpu"))
        context = [t.to(device) for t in context]

    return context

@torch.no_grad()
def build_context_list(wan_t2v, prompt_segments):
    """
    输入 prompt 配置列表，输出 context 列表。

    返回结构：
    [
        {
            "name": "seg_01_boxing",
            "type": "normal",
            "prompt": "...",
            "context": context_1,
            "frame_num": 49,
            "seed": 42,
        },
        ...
    ]
    """

    context_items = []

    for seg in prompt_segments:
        name = seg["name"]
        prompt = seg["prompt"]

        print(f"[Encode] {name}")
        print(f"Prompt: {prompt}")

        context = encode_prompt_to_context(wan_t2v, prompt)

        item = {
            "name": name,
            "type": "normal",
            "prompt": prompt,
            "context": context,
            "frame_num": seg.get("frame_num", 49),
            "seed": seg.get("seed", 42),
        }

        context_items.append(item)

    return context_items

@torch.no_grad()
def encode_negative_context(wan_t2v, negative_prompt):
    """
    生成 classifier-free guidance 里的无条件 / negative context。
    后面 generate(context_override=...) 时也可能需要传 context_null_override。
    """

    print("[Encode] negative prompt")
    print(f"Negative prompt: {negative_prompt}")

    context_null = encode_prompt_to_context(wan_t2v, negative_prompt)
    return context_null

def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    

    # The default sampling steps are 40 for image-to-video tasks and 50 for text-to-video tasks.
    if args.sample_steps is None:
        args.sample_steps = 50
        if "i2v" in args.task:
            args.sample_steps = 40

    if args.sample_shift is None:
        args.sample_shift = 5.0
        if "i2v" in args.task and args.size in ["832*480", "480*832"]:
            args.sample_shift = 3.0
        elif "flf2v" in args.task or "vace" in args.task:
            args.sample_shift = 16

    # The default number of frames are 1 for text-to-image tasks and 81 for other tasks.
    if args.frame_num is None:
        args.frame_num = 1 if "t2i" in args.task else 81

    # T2I frame_num check
    if "t2i" in args.task:
        assert args.frame_num == 1, f"Unsupport frame_num {args.frame_num} for task {args.task}"

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    # Size check
    assert args.size in SUPPORTED_SIZES[
        args.
        task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames to sample from a image or video. The number should be 4n+1"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="The size of the ring attention parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated image or video to.")
    parser.add_argument(
        "--src_video",
        type=str,
        default=None,
        help="The file of the source video. Default None.")
    parser.add_argument(
        "--src_mask",
        type=str,
        default=None,
        help="The file of the source mask. Default None.")
    parser.add_argument(
        "--src_ref_images",
        type=str,
        default=None,
        help="The file list of the source reference images. Separated by ','. Default None."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the image or video from.")
    parser.add_argument(
        "--use_prompt_extend",
        action="store_true",
        default=False,
        help="Whether to use prompt extend.")
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.")
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use.")
    parser.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="zh",
        choices=["zh", "en"],
        help="The target language of prompt extend.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="The seed to use for generating the image or video.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="[image to video] The image to generate the video from.")
    parser.add_argument(
        "--first_frame",
        type=str,
        default=None,
        help="[first-last frame to video] The image (first frame) to generate the video from."
    )
    parser.add_argument(
        "--last_frame",
        type=str,
        default=None,
        help="[first-last frame to video] The image (last frame) to generate the video from."
    )
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=5.0,
        help="Classifier free guidance scale.")
    parser.add_argument("--use_text_cross_cache", action="store_true")
    parser.add_argument("--text_cross_cache_frames", type=int, default=2)
    parser.add_argument("--text_cross_query_frames", type=int, default=1)
    parser.add_argument("--text_cross_strength", type=float, default=0.03)
    parser.add_argument("--text_cross_use_start_ratio", type=float, default=0.20)
    parser.add_argument("--text_cross_use_end_ratio", type=float, default=0.50)
    parser.add_argument("--text_cross_block_start", type=int, default=10)
    parser.add_argument("--text_cross_block_end", type=int, default=16)
    parser.add_argument("--use_kv_cache", action="store_true")
    parser.add_argument("--use_sink_kv_cache", action="store_true", default=False)
    parser.add_argument("--kv_cache_frames", type=int, default=2)
    parser.add_argument("--kv_cache_query_frames", type=int, default=1)
    parser.add_argument("--kv_cache_strength", type=float, default=0.005)
    parser.add_argument("--kv_cache_use_start_ratio", type=float, default=0.35)
    parser.add_argument("--kv_cache_use_end_ratio", type=float, default=0.65)
    parser.add_argument("--kv_block_start", type=int, default=13)
    parser.add_argument("--kv_block_end", type=int, default=17)
    parser.add_argument("--latent_overlap", type=int, default=2)
    parser.add_argument("--use_initial_state_cache", action="store_true", default=False)
    parser.add_argument("--initial_step_start", type=int, default=0)
    parser.add_argument("--initial_step_end", type=int, default=2)
    parser.add_argument("--initial_strength", type=float, default=0.1)
    args = parser.parse_args()

    _validate_args(args)

    return args


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1 or args.ring_size > 1
        ), f"context parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1 or args.ring_size > 1:
        assert args.ulysses_size * args.ring_size == world_size, f"The number of ulysses_size and ring_size should be equal to the world size."
        from xfuser.core.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size())

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )

    if args.use_prompt_extend:
        if args.prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(
                model_name=args.prompt_extend_model,
                is_vl="i2v" in args.task or "flf2v" in args.task)
        elif args.prompt_extend_method == "local_qwen":
            prompt_expander = QwenPromptExpander(
                model_name=args.prompt_extend_model,
                is_vl="i2v" in args.task,
                device=rank)
        else:
            raise NotImplementedError(
                f"Unsupport prompt_extend_method: {args.prompt_extend_method}")

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    if "t2v" in args.task or "t2i" in args.task:
        logging.info(f"Input prompt: {args.prompt}")
        if args.use_prompt_extend:
            logging.info("Extending prompt ...")
            if rank == 0:
                prompt_output = prompt_expander(
                    args.prompt,
                    tar_lang=args.prompt_extend_target_lang,
                    seed=args.base_seed)
                if prompt_output.status == False:
                    logging.info(
                        f"Extending prompt failed: {prompt_output.message}")
                    logging.info("Falling back to original prompt.")
                    input_prompt = args.prompt
                else:
                    input_prompt = prompt_output.prompt
                input_prompt = [input_prompt]
            else:
                input_prompt = [None]
            if dist.is_initialized():
                dist.broadcast_object_list(input_prompt, src=0)
            args.prompt = input_prompt[0]
            logging.info(f"Extended prompt: {args.prompt}")

        logging.info("Creating WanT2V pipeline.")
        wan_t2v = wan.WanT2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
        )
        logging.info("Encoding negative context ...")
        context_null = encode_negative_context(wan_t2v, NEGATIVE_PROMPT)
        logging.info("Building context list ...")
        context_items = build_context_list(wan_t2v, PROMPT_SEGMENTS)
        logging.info(f"Total generation items: {len(context_items)}")
        
        logging.info(
            f"Generating {'image' if 't2i' in args.task else 'video'} ...")
        
        final_video = None
        prev_kv_cache_cond = None
        sink_kv_cache_cond = None
        prev_text_cross_cache_cond = None
        prev_initial_state_cache_cond = None
        
        use_kv_cache = getattr(args, "use_kv_cache", False)
        use_sink_kv_cache = getattr(args, "use_sink_kv_cache", False)
        use_initial_state_cache = getattr(args, "use_initial_state_cache", False)

        kv_cache_frames = getattr(args, "kv_cache_frames", 2)
        kv_cache_query_frames = getattr(args, "kv_cache_query_frames", 2)
        kv_cache_strength = getattr(args, "kv_cache_strength", 2)
        kv_cache_use_start_ratio = getattr(args, "kv_cache_use_start_ratio", 0.01)
        kv_cache_use_end_ratio = getattr(args, "kv_cache_use_end_ratio", 0.31)
        kv_block_start = getattr(args, "kv_block_start", 14)
        kv_block_end = getattr(args, "kv_block_end", 18)
        latent_overlap = getattr(args, "latent_overlap", 0)
        
        initial_strength = getattr(args, "initial_strength", 0.1)
        
        initial_step_start = getattr(args, "initial_step_start", 0)
        initial_step_end = getattr(args, "initial_step_end", 2)
        
        for idx, item in enumerate(context_items):
            logging.info("=" * 80)
            logging.info(f"[Generate segment {idx + 1}/{len(context_items)}] {item['name']}")
            logging.info(f"Prompt: {item['prompt']}")
            logging.info(f"Seed: {item['seed']}")
            logging.info(f"Frame num: {item['frame_num']}")

            latent_frames = (item["frame_num"] - 1) // wan_t2v.vae_stride[0] + 1
            if use_kv_cache:
                latent_frame_offset = idx * (latent_frames - latent_overlap)
            else:
                latent_overlap = 0
                latent_frame_offset = 0
                
            return_head_kv_cache = (use_kv_cache and use_sink_kv_cache and idx == 0)
            return_initial_state_cache = (use_initial_state_cache and idx == 0)

            active_sink_kv_cache_cond = (
                sink_kv_cache_cond
                if (use_kv_cache and use_sink_kv_cache and idx >= 1)
                else None
            )
            active_initial_state_cache_cond = (
                prev_initial_state_cache_cond
                if (use_initial_state_cache and idx >= 1)
                else None
            )
            
            kv_cache_cond,text_cross_cache_cond,initial_state_cache_cond, cur_video = wan_t2v.generate(
                input_prompt="",
                size=SIZE_CONFIGS[args.size],
                frame_num=item["frame_num"],
                shift=args.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.sample_steps,
                guide_scale=args.sample_guide_scale,
                seed=item["seed"],
                offload_model=args.offload_model,

                context_override=item["context"],
                context_null_override=context_null,

# KV cache
            sink_kv_cache_cond=active_sink_kv_cache_cond,
            return_head_kv_cache=return_head_kv_cache,
            prev_kv_cache_cond=(
                prev_kv_cache_cond if use_kv_cache else None
            ),
            return_kv_cache=use_kv_cache,
            kv_cache_frames=kv_cache_frames,
            kv_cache_query_frames=kv_cache_query_frames,
            kv_cache_strength=kv_cache_strength,
            kv_cache_use_start_ratio=kv_cache_use_start_ratio,
            kv_cache_use_end_ratio=kv_cache_use_end_ratio,
            kv_block_start=kv_block_start,
            kv_block_end=kv_block_end,
            latent_frame_offset=latent_frame_offset,

                prev_text_cross_cache_cond=(
                    prev_text_cross_cache_cond
                    if args.use_text_cross_cache else None
                ),
                return_text_cross_cache=args.use_text_cross_cache,
                text_cross_cache_frames=args.text_cross_cache_frames,
                text_cross_query_frames=args.text_cross_query_frames,
                text_cross_strength=args.text_cross_strength,
                text_cross_use_start_ratio=args.text_cross_use_start_ratio,
                text_cross_use_end_ratio=args.text_cross_use_end_ratio,
                text_cross_block_start=args.text_cross_block_start,
                text_cross_block_end=args.text_cross_block_end,
                                # initial state cache
                prev_initial_state_cache_cond=active_initial_state_cache_cond,
                return_initial_state_cache=return_initial_state_cache,
                initial_step_start=initial_step_start,
                initial_step_end=initial_step_end,
                initial_strength=initial_strength
            )
            if use_initial_state_cache:
                if idx == 0:
                    prev_initial_state_cache_cond = initial_state_cache_cond
                    logging.info("[InitialStateCache] Saved first-segment initial state cache.")
                else:
                    logging.info("[InitialStateCache] Reused first-segment initial state cache.")
            else:
                prev_initial_state_cache_cond = None
                
                
            if use_kv_cache:
                if (use_sink_kv_cache and idx == 0 and kv_cache_cond is not None):
                    sink_kv_cache_cond = {}
                    for step_idx, step_cache in enumerate(kv_cache_cond):
                        if step_cache is None:
                            sink_kv_cache_cond[step_idx] = None
                            continue
                        sink_step_cache = {}
                        for block_idx, block_cache in step_cache.items():
                            if ("head_k" in block_cache and "head_v" in block_cache and "head_lens" in block_cache):
                                sink_step_cache[block_idx] = {
                                    "k": block_cache["head_k"],
                                    "v": block_cache["head_v"],
                                    "lens": block_cache["head_lens"],
                                }
                        sink_kv_cache_cond[step_idx] = (sink_step_cache if len(sink_step_cache) > 0 else None)
                        logging.info("[KVCache] Saved first-segment head KV as Sink cache.")
                prev_kv_cache_cond = kv_cache_cond
            else:
                prev_kv_cache_cond = None
                sink_kv_cache_cond = None
                prev_kv_cache_uncond = None
            if args.use_text_cross_cache:
                prev_text_cross_cache_cond = text_cross_cache_cond
            else:
                prev_text_cross_cache_cond = None

            if cur_video is None:
                raise RuntimeError(f"Segment {idx} returned None video.")

            if final_video is None:
                final_video = cur_video
            else:
                final_video = torch.cat([final_video, cur_video], dim=1)

        video = final_video

    

    if rank == 0:
        if args.save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = args.prompt.replace(" ", "_").replace("/",
                                                                     "_")[:50]
            suffix = '.png' if "t2i" in args.task else '.mp4'
            args.save_file = f"{args.task}_{args.size.replace('*','x') if sys.platform=='win32' else args.size}_{args.ulysses_size}_{args.ring_size}_{formatted_prompt}_{formatted_time}" + suffix

        if "t2i" in args.task:
            logging.info(f"Saving generated image to {args.save_file}")
            cache_image(
                tensor=video.squeeze(1)[None],
                save_file=args.save_file,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))
        else:
            logging.info(f"Saving generated video to {args.save_file}")
            cache_video(
                tensor=video[None],
                save_file=args.save_file,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))
    logging.info("Finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
