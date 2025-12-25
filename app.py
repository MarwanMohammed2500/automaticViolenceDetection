import streamlit as st
import tempfile
from utils.video_reader import read_video
from utils.inference import predict

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
            video_tensor = read_video(
                temp_file.name,
                target_num_frames=16,
                frame_rate=30,
                normalize=True
            )

            prob = predict(video_tensor)

            if prob > 0.5:
                st.error(f"🚨 Violence Detected\n\nConfidence: {prob:.2f}")
                # st.error(f"🚨 Violence Detected")
            else:
                st.success(f"✅ Non-Violent Activity\n\nConfidence: {1 - prob:.2f}")
                # st.success(f"✅ Non-Violent Activity")
