import streamlit as st
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av  # Library PyAV untuk menangani media frame bawaan streamlit-webrtc

# 1. Konfigurasi Halaman
st.set_page_config(
    page_title="Real-time Mask Detector",
    page_icon="😷",
    layout="centered"
)

st.title("😷 Real-time Face Mask Detector & Tracking")
st.write("Aplikasi menggunakan standar modern `recv()` untuk pelacakan wajah bermasker yang lebih stabil.")

# 2. Muat Model TFLite & Haar Cascade Wajah (Cached)
@st.cache_resource
def load_models():
    interpreter = tflite.Interpreter(model_path="model_masker.tflite")
    interpreter.allocate_tensors()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return interpreter, face_cascade

interpreter, face_cascade = load_models()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 3. Struktur Baru: Menggunakan VideoProcessorBase dan recv()
class FaceMaskProcessor(VideoProcessorBase):
    def __init__(self):
        self.interpreter = interpreter
        self.face_cascade = face_cascade
        self.input_details = input_details
        self.output_details = output_details

    def recv(self, frame):
        # 1. Konversi frame PyAV ke format NumPy image (BGR untuk OpenCV)
        img = frame.to_ndarray(format="bgr24")
        
        # 2. Deteksi wajah menggunakan Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        
        for (x, y, w, h) in faces:
            try:
                # Potong area wajah (ROI)
                roi_color = img[y:y+h, x:x+w]
                
                # Preprocessing: Resize ke 120x120 & konversi ke RGB
                roi_rgb = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(roi_rgb, (120, 120))
                
                # Normalisasi (1./255) & Expand dimensi -> (1, 120, 120, 3)
                img_array = np.array(img_resized, dtype=np.float32) / 255.0
                img_batch = np.expand_dims(img_array, axis=0)
                
                # Jalankan Prediksi TFLite
                self.interpreter.set_tensor(self.input_details[0]['index'], img_batch)
                self.interpreter.invoke()
                prediction = self.interpreter.get_tensor(self.output_details[0]['index'])[0][0]
                
                # Penentuan Label & Warna Kotak Pelacak
                if prediction < 0.5:
                    label = f"Bermasker: {(1 - prediction)*100:.1f}%"
                    color = (0, 255, 0) # Hijau
                else:
                    label = f"Tanpa Masker: {prediction*100:.1f}%"
                    color = (0, 0, 255) # Merah
                
                # Gambar bounding box dan teks di atas wajah
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 3)
                cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
            except Exception:
                pass
                
        # 3. Kembalikan array kembali dalam bentuk objek av.VideoFrame
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# 4. Jalankan WebRTC Streamer dengan parameter baru
st.subheader("Live Camera Feed")
webrtc_streamer(
    key="face-mask-detection-recv",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=FaceMaskProcessor, # Perubahan nama parameter dari video_transformer_factory
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)

st.info("💡 Aplikasi berjalan menggunakan PyAV engine yang memastikan performa tracking FPS jauh lebih lancar.")