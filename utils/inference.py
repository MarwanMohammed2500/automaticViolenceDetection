import torch
import joblib
import numpy as np

device = torch.device("cpu")

clf = joblib.load("models/violence_classifier_svm.joblib")

@torch.inference_mode()
def extract_features(model, video_tensor: torch.Tensor) -> np.ndarray:
    """
    Input: (C, T, H, W)
    Output: (1, N_features)
    """
    video_tensor = video_tensor.unsqueeze(0).to(device)  # (1, C, T, H, W)

    features = model(video_tensor)

    # لو الـ C3D رجع Tensor عالي الأبعاد
    features = features.view(features.size(0), -1)

    return features.cpu().numpy()


def predict(features: np.ndarray):
    """
    Input: (1, N_features)
    Output: class + confidence
    """
    label = clf.predict(features)[0]

    if hasattr(clf, "predict_proba"):
        prob = clf.predict_proba(features)[0][1]
    else:
        prob = float(label)

    return label, prob
