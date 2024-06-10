import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle

# Load trained SVM model
with open('svm_model.pkl', 'rb') as model_file:
    clf = pickle.load(model_file)

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

#Judul dan deskripsi
st.title("Sign Catcher")


# Panduan huruf di sidebar
st.sidebar.title("Panduan Huruf")
st.sidebar.image ("foto.jpeg", caption="Panduan Huruf Bahasa Isyarat", width=500 )

# Video capture dari webcam
run = st.checkbox('Jalankan Kamera')
FRAME_WINDOW = st.image([])

if run:
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.write("Tidak dapat membaca frame dari kamera.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Konversi ke RGB
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:  # Jika tangan terdeteksi
            for hand_landmarks in results.multi_hand_landmarks:
                # Gambar landmark
                for landmark in hand_landmarks.landmark:
                    height, width, _ = frame.shape
                    cx, cy = int(landmark.x * width), int(landmark.y * height)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)  # Warna merah untuk titik landmark
                
                # Gambar garis antara landmark
                for idx in range(1, len(hand_landmarks.landmark)):
                    prev_landmark = hand_landmarks.landmark[idx - 1]
                    cur_landmark = hand_landmarks.landmark[idx]
                    prev_cx, prev_cy = int(prev_landmark.x * width), int(prev_landmark.y * height)
                    cur_cx, cur_cy = int(cur_landmark.x * width), int(cur_landmark.y * height)
                    cv2.line(frame, (prev_cx, prev_cy), (cur_cx, cur_cy), (0, 255, 0), 2)  # Warna hijau untuk garis antara landmark
                
                # Gambar kotak di dalam hand landmark
                min_x, min_y, max_x, max_y = np.inf, np.inf, -np.inf, -np.inf
                for landmark in hand_landmarks.landmark:
                    height, width, _ = frame.shape
                    cx, cy = int(landmark.x * width), int(landmark.y * height)
                    if cx < min_x:
                        min_x = cx
                    if cy < min_y:
                        min_y = cy
                    if cx > max_x:
                        max_x = cx
                    if cy > max_y:
                        max_y = cy
                
                cv2.rectangle(frame, (min_x - 10, min_y - 10), (max_x + 10, max_y + 10), (0, 0, 255), 2)
                
                # Deteksi huruf
                hand_features = np.array([[lmk.x, lmk.y, lmk.z] for lmk in hand_landmarks.landmark]).flatten()
                prediction = clf.predict([hand_features])
                text = prediction[0]
                # Tulis huruf pada kotak dengan teks berwarna hitam
                cv2.putText(frame, text, (min_x - 5, min_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)
    
    cap.release()
