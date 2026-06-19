# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import logging
import os
import sys
import warnings
warnings.filterwarnings('ignore')
import random
import torch
import torch.distributed as dist

import wan
from wan.configs import SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import  cache_video, str2bool

from wan.configs import WAN_CONFIGS

from wan.utils.utils import cache_video

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
    parser.add_argument(
    "--use_velocity_interp",
    action="store_true",
    help="Use tail velocity cache from previous segment to guide the head of the next segment."
)

    parser.add_argument(
    "--velocity_overlap",
    type=int,
    default=4,
    help="Number of latent-frame positions used for velocity interpolation."
)

    parser.add_argument(
    "--velocity_alpha_start",
    type=float,
    default=1.0,
    help="Initial alpha for velocity interpolation."
)

    parser.add_argument(
    "--velocity_start_step_ratio",
    type=float,
    default=0.6,
    help="Use velocity interpolation in the first ratio of denoising steps."
)
    parser.add_argument(
    "--velocity_end_step_ratio",
    type=float,
    default=0.9,
    help="Stop using velocity interpolation after this ratio of denoising steps."
)
    
    parser.add_argument(
    "--noise_overlap",
    type=int,
    default=4,
    help="Number of latent frames whose initial noise is shared from previous segment tail to current segment head."
    )
    parser.add_argument(
    "--hidden_overlap",
    type=int,
    default=2,
    help="Number of latent time steps for hidden state tail-head fusion."
)

    parser.add_argument(
    "--hidden_alpha",
    type=float,
    default=0.03,
    help="Fusion alpha for hidden state tail-head fusion."
)

    parser.add_argument(
    "--hidden_use_step_ratio",
    type=float,
    default=0.4,
    help="Use hidden state fusion in the first ratio of denoising steps."
)

    parser.add_argument(
    "--hidden_block_start",
    type=int,
    default=8,
    help="Start block index for hidden state fusion."
)

    parser.add_argument(
    "--hidden_block_end",
    type=int,
    default=20,
    help="End block index, exclusive, for hidden state fusion."
)
    args = parser.parse_args()

    _validate_args(args)

    return args

PROMPT_SEGMENTS = [
    {
        "name": "seg_01",
        "prompt": "A steady medium shot of the red rose, where the camera continues retreating to show the full stem with its leaves and neighboring buds.",
        "frame_num": 49,
        "seed": 42,
    },
    {
        "name": "seg_02",
        "prompt": "A gentle close shot of a red rose petal, where the camera gradually pulls back to reveal the entire unfurling bloom in its perfect symmetry.",
        "frame_num": 49,
        "seed": 42,
    },
    {
        "name": "seg_03",
        "prompt": "A smooth full shot of the red rose bush, where the camera moves further back to encompass the entire garden bed and surrounding flowering plants.",
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

        print_context_shape(name, context)

    return context_items
def pad_context_to_same_length(c1, c2):
    if c1.shape == c2.shape:
        return c1, c2

    if c1.ndim != c2.ndim:
        raise ValueError(f"context ndim 不一致: {c1.shape} vs {c2.shape}")

    if c1.shape[1:] != c2.shape[1:]:
        raise ValueError(f"context hidden 维度不一致: {c1.shape} vs {c2.shape}")

    len1 = c1.shape[0]
    len2 = c2.shape[0]
    max_len = max(len1, len2)

    def pad_to_len(x, target_len):
        if x.shape[0] == target_len:
            return x

        pad_shape = (target_len - x.shape[0],) + x.shape[1:]
        pad_tensor = torch.zeros(
            pad_shape,
            dtype=x.dtype,
            device=x.device,
        )

        return torch.cat([x, pad_tensor], dim=0)

    return pad_to_len(c1, max_len), pad_to_len(c2, max_len)
def interpolate_conditions(c1, c2, weight_1, weight_2):
    c1, c2 = pad_context_to_same_length(c1, c2)
    return weight_1 * c1 + weight_2 * c2
def interpolate_wan_context(context_1, context_2, alpha):
    """
    Wan2.1 的 context 通常是 list[tensor]，
    所以对 list 里的每个 tensor 分别插值。
    """
    weight_1 = 1.0 - alpha
    weight_2 = alpha

    return [
        interpolate_conditions(c1, c2, weight_1, weight_2)
        for c1, c2 in zip(context_1, context_2)
    ]
def insert_transition_contexts(context_items, num_transition_blocks=1):
    """
    输入:
        [seg1, seg2, seg3]

    输出:
        [seg1, trans_01_02, seg2, trans_02_03, seg3]
    """
    if num_transition_blocks <= 0:
        return context_items

    new_items = []

    for i in range(len(context_items)):
        # 先加入当前正常段
        new_items.append(context_items[i])

        # 如果后面还有下一段，就插入 transition
        if i < len(context_items) - 1:
            cur_item = context_items[i]
            next_item = context_items[i + 1]

            
            alpha = 3/4

            transition_context = interpolate_wan_context(
                    cur_item["context"],
                    next_item["context"],
                    alpha
                )

            transition_item = {
                    "name": f"trans",
                    "type": "transition",
                    "prompt": None,
                    "from": cur_item["name"],
                    "to": next_item["name"],
                    "alpha": alpha,
                    "context": transition_context,
                    "frame_num": 33,
                    "seed": cur_item["seed"],
                }

            new_items.append(transition_item)

    return new_items
@torch.no_grad()
def encode_negative_context(wan_t2v, negative_prompt):
    """
    生成 classifier-free guidance 里的无条件 / negative context。
    后面 generate(context_override=...) 时也可能需要传 context_null_override。
    """

    print("[Encode] negative prompt")
    print(f"Negative prompt: {negative_prompt}")

    context_null = encode_prompt_to_context(wan_t2v, negative_prompt)

    print_context_shape("negative_context", context_null)

    return context_null
def print_context_shape(name, context):
    print(f"[Context Shape] {name}")

    if isinstance(context, list):
        for i, t in enumerate(context):
            print(f"  context[{i}]: shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}")
    elif torch.is_tensor(context):
        print(f"  tensor: shape={tuple(context.shape)}, dtype={context.dtype}, device={context.device}")
    else:
        print(f"  type={type(context)}")

    print("-" * 60)

#参数合法性校验
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

    # ============================================================
    # 1. 基础初始化
    # ============================================================
    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}."
        )

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size
        )
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), "t5_fsdp and dit_fsdp are not supported in non-distributed environments."

        assert not (
            args.ulysses_size > 1 or args.ring_size > 1
        ), "context parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1 or args.ring_size > 1:
        assert args.ulysses_size * args.ring_size == world_size, (
            "The number of ulysses_size and ring_size should be equal to the world size."
        )

        from xfuser.core.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )

        init_distributed_environment(
            rank=dist.get_rank(),
            world_size=dist.get_world_size()
        )

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )

    # ============================================================
    # 2. prompt extend，保留原逻辑
    # ============================================================
    if args.use_prompt_extend:
        if args.prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(
                model_name=args.prompt_extend_model,
                is_vl="i2v" in args.task or "flf2v" in args.task
            )
        elif args.prompt_extend_method == "local_qwen":
            prompt_expander = QwenPromptExpander(
                model_name=args.prompt_extend_model,
                is_vl="i2v" in args.task,
                device=rank
            )
        else:
            raise NotImplementedError(
                f"Unsupport prompt_extend_method: {args.prompt_extend_method}"
            )

    cfg = WAN_CONFIGS[args.task]

    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, (
            f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."
        )

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    # ============================================================
    # 3. 只处理 t2v
    # ============================================================
    if "t2v" not in args.task:
        raise ValueError(f"Unknown task type: {args.task}")

    if args.prompt is None:
        args.prompt = PROMPT_SEGMENTS[0]["prompt"]

    logging.info(f"Input prompt: {args.prompt}")

    if args.use_prompt_extend:
        logging.info("Extending prompt ...")

        if rank == 0:
            prompt_output = prompt_expander(
                args.prompt,
                tar_lang=args.prompt_extend_target_lang,
                seed=args.base_seed
            )

            if prompt_output.status == False:
                logging.info(f"Extending prompt failed: {prompt_output.message}")
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

    # ============================================================
    # 4. 创建 WanT2V
    # ============================================================
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

    # ============================================================
    # 5. 构造 prompt/context 列表
    #    保留 prompt/context 线性插值
    # ============================================================
    logging.info("Encoding negative context ...")
    context_null = encode_negative_context(wan_t2v, NEGATIVE_PROMPT)

    logging.info("Building context list ...")
    context_items = build_context_list(wan_t2v, PROMPT_SEGMENTS)

    #context_items = insert_transition_contexts(
        #context_items,
        #num_transition_blocks=getattr(args, "num_transition_blocks", 2)
    #)

    logging.info(f"Total generation items: {len(context_items)}")

    # ============================================================
    # 6. 多段生成状态
    # 现在保存的是 decoded video
    # ============================================================
    final_video = None
    prev_velocity_tail_cache = None
    prev_noise_tail_cache = None
    prev_hidden_tail_cache = None

    use_velocity_interp = getattr(args, "use_velocity_interp", True)
    velocity_overlap = getattr(args, "velocity_overlap", 4)
    velocity_alpha_start = getattr(args, "velocity_alpha_start", 0.4)
    velocity_start_step_ratio = getattr(args, "velocity_start_step_ratio", 0.6)
    velocity_end_step_ratio = getattr(args, "velocity_end_step_ratio", 0.9)

    hidden_overlap = getattr(args, "hidden_overlap", 2)
    hidden_alpha = getattr(args, "hidden_alpha", 0.03)
    hidden_use_step_ratio = getattr(args, "hidden_use_step_ratio", 0.4)
    hidden_block_start = getattr(args, "hidden_block_start", 8)
    hidden_block_end = getattr(args, "hidden_block_end", 20)
    
    noise_overlap = getattr(args, "noise_overlap", 4)

    # ============================================================
    # 7. 逐段生成
    # ============================================================
    for idx, item in enumerate(context_items):
        logging.info("=" * 80)
        logging.info(f"[Generate segment {idx + 1}/{len(context_items)}] {item['name']}")
        logging.info(f"Type: {item.get('type', 'normal')}")

        if item.get("type") == "transition":
            logging.info(f"Transition: {item['from']} -> {item['to']}")
            logging.info(f"Alpha: {item['alpha']}")
        else:
            logging.info(f"Prompt: {item['prompt']}")

        logging.info(f"Seed: {item['seed']}")
        logging.info(f"Frame num: {item['frame_num']}")

        # ------------------------------------------------------------
        # 现在 WanT2V.generate 返回：
        #   velocity_tail_cache, latent_tail_cache, cur_video
        #
        # cur_video.shape 预计是：
        #   [C, T, H, W]
        # ------------------------------------------------------------
        velocity_tail_cache, noise_tail_cache,hidden_tail_cache, cur_video = wan_t2v.generate(
            input_prompt="",
            size=SIZE_CONFIGS[args.size],
            frame_num=item["frame_num"],
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=item["seed"],
            offload_model=args.offload_model,

            # prompt/context 线性插值
            context_override=item["context"],
            context_null_override=context_null,

            # velocity 插值
            return_velocity_cache=use_velocity_interp,
            prev_velocity_tail_cache=(
                prev_velocity_tail_cache if use_velocity_interp else None
            ),
            velocity_overlap=velocity_overlap,
            velocity_alpha_start=velocity_alpha_start,
            velocity_start_step_ratio=velocity_start_step_ratio,
            velocity_end_step_ratio=velocity_end_step_ratio,
           
            # noise 共享
            return_noise_tail_cache=True,
            prev_noise_tail_cache=prev_noise_tail_cache,
            noise_overlap=noise_overlap,
            
            # hidden state fusion
            return_hidden_tail_cache=True,
            prev_hidden_tail_cache=prev_hidden_tail_cache,
            hidden_overlap=hidden_overlap,
            hidden_alpha=hidden_alpha,
            hidden_use_step_ratio=hidden_use_step_ratio,
            hidden_block_start=hidden_block_start,
            hidden_block_end=hidden_block_end,
        )

      
        if use_velocity_interp:
            prev_velocity_tail_cache = velocity_tail_cache
        else:
            prev_velocity_tail_cache = None
        prev_noise_tail_cache = noise_tail_cache
        prev_hidden_tail_cache = hidden_tail_cache
        # ------------------------------------------------------------
        # 非 rank0 不负责保存视频
        # ------------------------------------------------------------
        if rank != 0:
            continue

        if cur_video is None:
            raise RuntimeError(f"Segment {idx} returned None video on rank 0.")

        logging.info(f"[Video] current video shape = {tuple(cur_video.shape)}")

        # ------------------------------------------------------------
        # 直接拼 decoded video
        #
        # 一般 Wan VAE decode 出来是 [C, T, H, W]
        # 所以沿 dim=1 拼接。
        #
        # 注意：
        # 不再做 latent blend
        # 不再做 append_latent_with_overlap
        # ------------------------------------------------------------
        if final_video is None:
            final_video = cur_video
        else:
            final_video = torch.cat([final_video, cur_video], dim=1)

        logging.info(f"[Final video] current shape = {tuple(final_video.shape)}")

    # ============================================================
    # 8. 保存最终视频
    # ============================================================
    if rank == 0:
        if final_video is None:
            raise RuntimeError("No video was generated.")

        if args.save_file is None:
            args.save_file = "experiment/velocity_interp_prompt_interp.mp4"

        save_dir = os.path.dirname(args.save_file)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        logging.info(f"Saving final video to {args.save_file}")

        cache_video(
            tensor=final_video[None],
            save_file=args.save_file,
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )

        logging.info("Finished.")
        return
    



if __name__ == "__main__":
    args = _parse_args()
    generate(args)
