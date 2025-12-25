import streamlit as st
import tempfile
import json
import cv2
import torch
from utils.video_loader import load_video
from utils.inference import predict

st.set_page_config(
    page_title="Violence Detection System",
    layout="wide"
)

st.title("🚨 Automatic Violence Detection in Videos")

tab1, tab2, tab3 = st.tabs([
    "📤 Upload Video",
    "🎥 Real-Time Detection",
    "📊 Model Performance"
])

# ======================================================
# TAB 1 – Upload Video
# ======================================================
with tab1:
    st.header("Upload a Video for Violence Detection")

    uploaded_video = st.file_uploader(
        "Upload video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_video.read())

        st.video(uploaded_video)

        if st.button("Analyze Video"):
            with st.spinner("Analyzing..."):
                video_tensor = load_video(temp_file.name)

                if video_tensor is None:
                    st.warning("Video too short. Minimum 16 frames required.")
                else:
                    prob = predict(video_tensor)

                    if prob > 0.5:
                        st.error(f"🚨 Violence Detected — Confidence: {prob:.2f}")
                    else:
                        st.success(f"✅ Non-Violent — Confidence: {1 - prob:.2f}")

# ======================================================
# TAB 2 – Real-Time Detection
# ======================================================
with tab2:
    st.header("Real-Time Violence Detection (Webcam)")

    run = st.checkbox("Start Camera")

    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)

    buffer = []

    while run:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb)

        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1) / 255.0
        frame_tensor = torch.nn.functional.interpolate(
            frame_tensor.unsqueeze(0),
            size=(224, 224),
            mode="bilinear"
        ).squeeze(0)

        buffer.append(frame_tensor)

        if len(buffer) == 16:
            video_tensor = torch.stack(buffer)
            buffer.clear()

            prob = predict(video_tensor)

            if prob > 0.5:
                st.error("🚨 Violence Detected!")
            else:
                st.success("✅ Normal Activity")

    cap.release()

# ======================================================
# TAB 3 – Model Accuracy
# ======================================================
with tab3:
    st.header("Model Evaluation Results")

    try:
        with open("results/metrics.json") as f:
            metrics = json.load(f)

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
        col2.metric("Precision", f"{metrics['precision']:.2f}")
        col3.metric("Recall", f"{metrics['recall']:.2f}")
        col4.metric("F1-Score", f"{metrics['f1_score']:.2f}")

        st.subheader("Confusion Matrix")
        st.image("results/confusion_matrix.png")

    except Exception as e:
        st.warning("Evaluation results not available.")
