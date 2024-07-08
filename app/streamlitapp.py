# Import necessary libraries
import streamlit as st
import os
import imageio
import numpy as np
import tensorflow as tf
from moviepy.editor import VideoFileClip
from utils import load_data, num_to_char
from modelutil import load_model

# Set the layout to the Streamlit app as wide
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar:
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipBuddy')
    st.info('This application is originally developed from the LipNet deep learning model.')

st.title('LipNet Full Stack App')

# Generating a list of options or videos
options = os.listdir(os.path.join('..', 'data', 's1'))
selected_video = st.selectbox('Choose video', options)

# Generate two columns
col1, col2 = st.columns(2)

if options:
    # Rendering the video
    with col1:
        st.info('The video below displays the converted video in mp4 format')
        file_path = os.path.join('..', 'data', 's1', selected_video)

        # Convert the video to mp4 format using moviepy
        video_clip = VideoFileClip(file_path)
        video_clip.write_videofile("test_video.mp4", codec="libx264", audio=False)

        # Display the video inside the app
        video_file = open('test_video.mp4', 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)

    with col2:
        st.info('This is all the machine learning model sees when making a prediction')

        # Load video frames using load_data function
        video, annotations = load_data(tf.convert_to_tensor(file_path))
        
        # Convert TensorFlow tensor to NumPy array if necessary
        if tf.is_tensor(video):
            video = video.numpy()

        # Check if video frames are valid
        if video.ndim == 4:  # Assuming (frames, height, width, channels)
            batch_size = 1
            time_steps = video.shape[0]  # Number of frames
            height, width, channels = video.shape[1], video.shape[2], video.shape[3]
            features = height * width * channels  # Flattened frame size

            # Normalize video frames
            video = video / 255.0

            # Reshape video data to (batch_size, time_steps, features)
            video = video.reshape((batch_size, time_steps, features))

            # Debugging: print the shape of the video array after reshaping
            st.write(f"Video array shape after reshaping: {video.shape}")

            # Load the model and predict
            model = load_model()
            yhat = model.predict(video)
            st.text(tf.argmax(yhat, axis=1))
            decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
            st.text(decoder)

            # Convert prediction to text
            st.info('Decode the raw tokens into words')
            converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
            st.text(converted_prediction)
        else:
            st.error("The video data does not have the expected number of dimensions.")

        # Save frames as a GIF
        if video.ndim == 4:
            # Rescale frames to uint8
            frames_uint8 = (video * 255).astype(np.uint8)
            # Ensure frames are in the correct format
            valid_frames = []
            for frame in frames_uint8:
                if frame.shape[2] == 1:  # If grayscale, convert to RGB
                    frame = np.repeat(frame, 3, axis=2)
                valid_frames.append(frame)

            imageio.mimsave('animation.gif', valid_frames, fps=10)
            st.image('animation.gif', width=400)
        else:
            st.error("No valid frames to save as GIF.")
