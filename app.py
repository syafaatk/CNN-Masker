import streamlit as st
import cv2
import numpy as np
import tflite-runtime.interpreter as tflite
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# 1. Konfigurasi Halaman
st.set_page_config(
    page_title="Real-time Mask Detector",
    page_icon="😷",
    layout="centered"
)

st.title("😷 Real-time Face Mask Detector & Tracking")
st.write("Aplikasi mendeteksi penggunaan masker secara real-time dengan tracking kotak pada wajah.")

# 2. Muat Model TFLite & Haar Cascade Wajah (Cached agar ringan)
@st.cache_resource
def load_models():
    # Load TFLite Model
    interpreter = tflite.Interpreter(model_path="model_masker.tflite")
    interpreter.allocate_tensors()
    
    # Load Pre-trained Face Detector bawaan OpenCV (Haar Cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    return interpreter, face_cascade

interpreter, face_cascade = load_models()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 3. Kelas Transformer untuk Memproses Video Stream per Frame
class FaceMaskTransformer(VideoTransformerBase):
    def __init__(self):
        self.interpreter = interpreter
        self.face_cascade = face_cascade
        self.input_details = input_details
        self.output_details = output_details

    def transform(self, frame):
        # Konversi frame video menjadi format numpy array (BGR)
        img = frame.to_ndarray(format="bgr24")
        
        # Konversi ke Grayscale untuk deteksi wajah yang lebih cepat
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Deteksi Wajah
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        
        for (x, y, w, h) in faces:
            try:
                # Potong area wajah (ROI)
                roi_color = img[y:y+h, x:x+w]
                
                # Preprocessing Wajah untuk Model (Resize ke 120x120 & konversi ke RGB)
                roi_rgb = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(roi_rgb, (120, 120))
                
                # Normalisasi (1./255) & Expand dimensi menjadi (1, 120, 120, 3)
                img_array = np.array(img_resized, dtype=np.float32) / 255.0
                img_batch = np.expand_dims(img_array, axis=0)
                
                # Jalankan Prediksi TFLite
                self.interpreter.set_tensor(self.input_details[0]['index'], img_batch)
                self.interpreter.invoke()
                prediction = self.interpreter.get_tensor(self.output_details[0]['index'])[0][0]
                
                # Tentukan Label dan Warna Kotak
                # Catatan: Jika saat training 0 = Masker dan 1 = Tanpa Masker
                if prediction < 0.5:
                    label = f"Bermasker: {(1 - prediction)*100:.1f}%"
                    color = (0, 255, 0) # Hijau (BGR)
                else:
                    label = f"Tanpa Masker: {prediction*100:.1f}%"
                    color = (0, 0, 255) # Merah (BGR)
                
                # Gambar kotak pelacak (Green/Red Box) pada wajah
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 3)
                
                # Tulis teks label di atas kotak wajah
                cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
            except Exception as e:
                pass
                
        return img

# 4. Jalankan Kamera Real-time di Streamlit
st.subheader("Live Camera Feed")
webrtc_streamer(
    key="face-mask-detection",
    mode=WebRtcMode.SENDRECV,
    video_transformer_factory=FaceMaskTransformer,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}] # Menggunakan server STUN Google agar lancar di Cloud
    },
    media_stream_constraints={"video": True, "audio": False}, # Matikan audio agar hemat bandwidth
)

st.info("💡 Catatan: Berikan izin akses kamera pada browser Anda. Kotak otomatis melacak wajah dan berubah warna berdasarkan status masker.")