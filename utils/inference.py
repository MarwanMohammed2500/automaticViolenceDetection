import torch

device = torch.device("cpu")

model = torch.load(
    "model/violenceDetectionModel.pt",
    map_location=device
)
model.eval()

@torch.no_grad()
def predict(video_tensor: torch.Tensor) -> float:
    """
    Input shape: (C, T, H, W)
    Returns: violence probability
    """
    video_tensor = video_tensor.unsqueeze(0)  # (1, C, T, H, W)
    output = model(video_tensor)

    # Binary classifier
    if output.shape[-1] == 1:
        prob = torch.sigmoid(output).item()
    else:
        prob = torch.softmax(output, dim=1)[0][1].item()

    return prob
