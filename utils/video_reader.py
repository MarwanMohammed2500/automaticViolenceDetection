import torch
import random
from typing import List
from torchvision.io import VideoReader
from torchvision.transforms.transforms import Resize

device = torch.device("cpu")

def read_video(
    video_path: str,
    target_num_frames: int = 16,
    frame_rate: int = 30,
    normalize: bool = True
) -> torch.Tensor:
    """
    Output shape: (C, T, H, W)
    """

    frames = []
    resizer = Resize((112, 112))
    video_reader = VideoReader(video_path, "video")

    for frame in video_reader:
        frames.append(frame["data"])

    num_frames = len(frames)
    num_seconds = num_frames / frame_rate
    sampling_step = max(1, round((frame_rate * num_seconds) / target_num_frames))

    sampled_frames = frames[::sampling_step]

    if len(sampled_frames) > target_num_frames:
        sampled_frames = sampled_frames[:target_num_frames]

    elif len(sampled_frames) < target_num_frames:
        diff = target_num_frames - len(sampled_frames)
        mid = len(sampled_frames) // 2
        idx = random.choice([
            max(mid - 1, 0),
            mid,
            min(mid + 1, len(sampled_frames) - 1)
        ])
        sampled_frames.extend([sampled_frames[idx]] * diff)

    processed_frames = []
    for frame in sampled_frames:
        frame = frame.float()
        if normalize:
            frame = frame / 255.0
        processed_frames.append(resizer(frame))

    video_tensor = torch.stack(processed_frames).permute(1, 0, 2, 3)
    return video_tensor.to(device)
