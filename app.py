import torch
import streamlit as st
import tempfile
from utils.video_reader import read_video
from utils.inference import predict, extract_features

# Load feature extractor
model = torch.jit.load("models/C3D_feature_extractor.pt")
model.eval()

st.set_page_config(
    page_title="Violence Detection System",
    layout="centered"
)

st.title("🚨 Automatic Violence Detection in Videos")
st.write("Upload a video and detect violent behavior using a deep learning model.")

uploaded_video = st.file_uploader(
    "Upload a video file",
    type=["mp4", "avi", "mov"]
)

if uploaded_video:
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_video.read())

    st.video(uploaded_video)

    if st.button("Analyze Video"):
        with st.spinner("Analyzing video..."):
            # Read video
            video_tensor = read_video(
                temp_file.name,
                target_num_frames=16,
                frame_rate=30,
                normalize=True
            )

            # Extract features
            features = extract_features(model, video_tensor)

            # Predict
            label, prob = predict(features)

            if label == 1:
                st.error(f"🚨 Violence Detected\n\nConfidence: {prob:.2f}")
            else:
                st.success(f"✅ Non-Violent Activity\n\nConfidence: {1 - prob:.2f}")
