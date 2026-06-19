import os
import cv2
import numpy as np


def read_video(video_path, target_size=None):
    """
    读取视频为 numpy 数组: [T, H, W, C]
    OpenCV 读出来是 BGR，这里保持 BGR，不影响拼接和保存。
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"找不到视频文件: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n读取视频: {video_path}")
    print(f"fps={fps}, width={width}, height={height}, frames={frame_count}")

    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if target_size is not None:
            target_width, target_height = target_size
            if width != target_width or height != target_height:
                frame = cv2.resize(frame, (target_width, target_height))

        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"视频没有读取到任何帧: {video_path}")

    return np.stack(frames, axis=0), fps


def write_video(output_path, frames, fps):
    """
    保存 numpy 视频数组: [T, H, W, C]
    """
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    frames = frames.astype(np.uint8)
    height, width = frames.shape[1], frames.shape[2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        writer.write(frame)

    writer.release()

    print("\n========== 保存完成 ==========")
    print(f"输出文件: {output_path}")
    print(f"输出帧数: {len(frames)}")
    print(f"fps: {fps}")
    print(f"size: {width}x{height}")


def linear_blend(old_overlap, new_overlap):
    """
    old_overlap: [K, H, W, C]，上一段末尾 K 帧
    new_overlap: [K, H, W, C]，下一段开头 K 帧

    返回:
    blended: [K, H, W, C]
    """
    if old_overlap.shape != new_overlap.shape:
        raise ValueError(
            f"overlap shape 不一致: old={old_overlap.shape}, new={new_overlap.shape}"
        )

    K = old_overlap.shape[0]

    old = old_overlap.astype(np.float32)
    new = new_overlap.astype(np.float32)

    # alpha 从 1 到 0：
    # 第一帧更接近 old，最后一帧更接近 new
    alpha = np.linspace(1.0, 0.0, K, dtype=np.float32)
    alpha = alpha[:, None, None, None]

    blended = alpha * old + (1.0 - alpha) * new
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return blended


def concat_with_overlap_blending(videos, overlap=16):
    """
    videos: list[np.ndarray]，每个元素形状 [T, H, W, C]
    overlap: 重叠帧数

    拼接逻辑:
    final = first_video
    对每个后续 video:
        final = final[:-overlap] + blend(final[-overlap:], video[:overlap]) + video[overlap:]
    """
    if len(videos) == 0:
        raise ValueError("videos 不能为空")

    final = videos[0]

    for i, video in enumerate(videos[1:], start=2):
        if len(final) < overlap:
            raise ValueError(f"当前 final 帧数 {len(final)} 小于 overlap={overlap}")
        if len(video) < overlap:
            raise ValueError(f"第 {i} 段视频帧数 {len(video)} 小于 overlap={overlap}")

        print(f"\n========== 融合第 {i - 1} 段和第 {i} 段 ==========")
        print(f"当前 final 帧数: {len(final)}")
        print(f"新视频帧数: {len(video)}")
        print(f"overlap: {overlap}")

        old_overlap = final[-overlap:]
        new_overlap = video[:overlap]

        blended = linear_blend(old_overlap, new_overlap)

        final = np.concatenate(
            [
                final[:-overlap],
                blended,
                video[overlap:],
            ],
            axis=0,
        )

        print(f"融合后 final 帧数: {len(final)}")

    return final


def main():
    video_paths = [
        "seg01.mp4",
        "seg02.mp4",
        "seg03.mp4",
    ]

    output_path = "experiment/final_overlap_blend.mp4"

    overlap = 16

    # 先读第一个视频，确定统一尺寸和 fps
    first_video, fps = read_video(video_paths[0])
    height, width = first_video.shape[1], first_video.shape[2]
    target_size = (width, height)

    videos = [first_video]

    # 后续视频统一 resize 到第一段视频尺寸
    for path in video_paths[1:]:
        video, cur_fps = read_video(path, target_size=target_size)

        if abs(cur_fps - fps) > 1e-3:
            print(f"警告: {path} 的 fps={cur_fps}，与第一段 fps={fps} 不一致。输出将使用第一段 fps。")

        videos.append(video)

    final = concat_with_overlap_blending(videos, overlap=overlap)

    write_video(output_path, final, fps=fps)


if __name__ == "__main__":
    main()