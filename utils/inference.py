import torch

device = torch.device("cpu")

model = torch.load("model/violence_model.pth", map_location=device)  ## the path of the model
model.eval()

@torch.no_grad()
def predict(video_tensor):
    video_tensor = video_tensor.unsqueeze(0)  # (1, T, C, H, W)
    output = model(video_tensor)

    if output.shape[-1] == 1:
        prob = torch.sigmoid(output).item()
    else:
        prob = torch.softmax(output, dim=1)[0][1].item()

    return prob
