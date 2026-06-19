import os
import cv2


def read_and_write_video(
    video_path,
    writer,
    target_width,
    target_height,
    drop_first_n_frames=0,
):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"找不到视频文件: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n处理视频: {video_path}")
    print(f"原参数: fps={fps}, width={width}, height={height}, frames={frame_count}")
    print(f"丢弃前 {drop_first_n_frames} 帧")

    frame_idx = 0
    written = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 丢弃前 N 帧
        if frame_idx < drop_first_n_frames:
            frame_idx += 1
            continue

        # 如果尺寸不同，强制 resize 到第一个视频尺寸
        if width != target_width or height != target_height:
            frame = cv2.resize(frame, (target_width, target_height))

        writer.write(frame)

        frame_idx += 1
        written += 1

    cap.release()
    print(f"写入帧数: {written}")
    return written


def concat_drop_overlap(video_paths, output_path, overlap=16):
    if len(video_paths) < 1:
        raise ValueError("video_paths 不能为空")

    # 读取第一个视频参数，作为输出视频参数
    first_video = video_paths[0]
    if not os.path.exists(first_video):
        raise FileNotFoundError(f"找不到第一个视频: {first_video}")

    cap0 = cv2.VideoCapture(first_video)
    if not cap0.isOpened():
        raise RuntimeError(f"无法打开第一个视频: {first_video}")

    fps = cap0.get(cv2.CAP_PROP_FPS)
    width = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT))
    cap0.release()

    print("========== 输出视频参数 ==========")
    print(f"fps={fps}, width={width}, height={height}")
    print(f"第一个视频帧数: {frame_count}")
    print(f"overlap={overlap}")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_written = 0

    for i, path in enumerate(video_paths):
        if i == 0:
            # 第一段完整保留
            drop_n = 0
        else:
            # 后续段丢掉前 overlap 帧
            drop_n = overlap

        written = read_and_write_video(
            video_path=path,
            writer=writer,
            target_width=width,
            target_height=height,
            drop_first_n_frames=drop_n,
        )
        total_written += written

    writer.release()

    print("\n========== 拼接完成 ==========")
    print(f"输出文件: {output_path}")
    print(f"总写入帧数: {total_written}")


if __name__ == "__main__":
    video_paths = [
        "seg01.mp4",
        "seg02.mp4",
        "seg03.mp4",
    ]

    output_path = "experiment/final_drop_overlap.mp4"

    overlap = 16

    concat_drop_overlap(video_paths, output_path, overlap=overlap)