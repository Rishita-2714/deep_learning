import streamlit as st
import os
import tempfile
from pathlib import Path
import subprocess

# Title of the app
st.markdown('<h1 style="color:lightgreen;">Game Sense</h1>', unsafe_allow_html=True)
st.markdown('<h2 style="color:lightgreen;">By Team Brahma_Hackers</h2>', unsafe_allow_html=True)

# Upload video
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Create a temporary directory to save the uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    st.video(temp_file_path)

    # Path to main.py
    main_script_path = "main.py"

    # Run main.py with the uploaded video file
    if st.button("Process Video"):
        # Display spinner while processing
        with st.spinner("Processing your video... It will take around 16 minutes for 10 sec video"):
            # Path to save the processed video
            result_video_path = "output_video/processed_video.avi"

            # Run the video processing script
            subprocess.run(
                ["python", main_script_path, temp_file_path, result_video_path],
                check=True
            )

        # Check if the processed video exists and display it
        if os.path.exists(result_video_path):
            st.success("Video processing complete!")

            # Provide a download link for the processed video
            with open(result_video_path, "rb") as file:
                st.download_button(
                    label="Download Processed Video",
                    data=file,
                    file_name="processed_video.avi",
                    mime="video/x-msvideo",
                )
        else:
            st.error("There was an issue with processing the video.")

    # Clean up temporary file
    os.remove(temp_file_path)
