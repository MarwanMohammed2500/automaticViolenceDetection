import av
import torch
import torchvision.transforms as T

NUM_FRAMES = 16
IMG_SIZE = 224

transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
])

def load_video(video_path, max_frames=NUM_FRAMES):
    container = av.open(video_path)
    frames = []

    for frame in container.decode(video=0):
        img = frame.to_image()
        img = transform(img)
        frames.append(img)

        if len(frames) == max_frames:
            break

    container.close()

    if len(frames) < max_frames:
        return None

    video_tensor = torch.stack(frames)  # (T, C, H, W)
    return video_tensor
