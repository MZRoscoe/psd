import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load trained SVM model
with open('svm_model.pkl', 'rb') as model_file:
    clf = pickle.load(model_file)

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Title and description
st.title("Sign Catcher")

# Guide image in the sidebar
st.sidebar.title("Panduan Huruf")
st.sidebar.image("foto.jpeg", caption="Panduan Huruf Bahasa Isyarat", width=300)

# Define the Video Transformer
class SignLanguageTransformer(VideoTransformerBase):
    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")  # Convert frame to BGR format
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for MediaPipe

        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:  # If hands are detected
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                for landmark in hand_landmarks.landmark:
                    height, width, _ = image.shape
                    cx, cy = int(landmark.x * width), int(landmark.y * height)
                    cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)  # Red color for landmark points
                
                # Draw lines between landmarks
                for idx in range(1, len(hand_landmarks.landmark)):
                    prev_landmark = hand_landmarks.landmark[idx - 1]
                    cur_landmark = hand_landmarks.landmark[idx]
                    prev_cx, prev_cy = int(prev_landmark.x * width), int(prev_landmark.y * height)
                    cur_cx, cur_cy = int(cur_landmark.x * width), int(cur_landmark.y * height)
                    cv2.line(image, (prev_cx, prev_cy), (cur_cx, cur_cy), (0, 255, 0), 2)  # Green color for lines
                
                # Draw a bounding box around the hand landmarks
                min_x, min_y, max_x, max_y = np.inf, np.inf, -np.inf, -np.inf
                for landmark in hand_landmarks.landmark:
                    height, width, _ = image.shape
                    cx, cy = int(landmark.x * width), int(landmark.y * height)
                    if cx < min_x:
                        min_x = cx
                    if cy < min_y:
                        min_y = cy
                    if cx > max_x:
                        max_x = cx
                    if cy > max_y:
                        max_y = cy
                
                cv2.rectangle(image, (min_x - 10, min_y - 10), (max_x + 10, max_y + 10), (0, 0, 255), 2)
                
                # Predict the letter
                hand_features = np.array([[lmk.x, lmk.y, lmk.z] for lmk in hand_landmarks.landmark]).flatten()
                prediction = clf.predict([hand_features])
                text = prediction[0]
                
                # Display the detected letter
                cv2.putText(image, text, (min_x - 5, min_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert image back to RGB for display

# Video stream using streamlit-webrtc
webrtc_streamer(key="sign-language-detector", video_processor_factory=SignLanguageTransformer)

# Cleanup MediaPipe hands object on app closure
hands.close()
