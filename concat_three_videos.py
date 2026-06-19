import os
import cv2


def concat_three_videos(video1_path, video2_path, video3_path, output_path):
    video_paths = [video1_path, video2_path, video3_path]

    # 检查文件是否存在
    for path in video_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到视频文件: {path}")

    # 先打开第一个视频，读取输出视频的基准参数
    cap0 = cv2.VideoCapture(video1_path)
    if not cap0.isOpened():
        raise RuntimeError(f"无法打开视频: {video1_path}")

    fps = cap0.get(cv2.CAP_PROP_FPS)
    width = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap0.release()

    print(f"输出视频参数: fps={fps}, width={width}, height={height}")

    # 选择编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_written = 0

    for idx, path in enumerate(video_paths, start=1):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {path}")

        cur_fps = cap.get(cv2.CAP_PROP_FPS)
        cur_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cur_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\n正在处理第 {idx} 段视频: {path}")
        print(f"原参数: fps={cur_fps}, width={cur_width}, height={cur_height}, frames={frame_count}")

        written_this_video = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 如果尺寸不同，强制 resize 到第一个视频的尺寸
            if cur_width != width or cur_height != height:
                frame = cv2.resize(frame, (width, height))

            writer.write(frame)
            written_this_video += 1
            total_written += 1

        cap.release()
        print(f"第 {idx} 段写入完成，写入帧数: {written_this_video}")

    writer.release()
    print(f"\n拼接完成！")
    print(f"输出文件: {output_path}")
    print(f"总写入帧数: {total_written}")


if __name__ == "__main__":
    video1 = "seg01.mp4"
    video2 = "seg02.mp4"
    video3 = "seg03.mp4"
    output = "experiment/final_concat.mp4"

    os.makedirs(os.path.dirname(output), exist_ok=True)

    concat_three_videos(video1, video2, video3, output)