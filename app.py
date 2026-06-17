import streamlit as st
import cv2
import numpy as np
from PIL import Image

# 1. Konfigurasi Halaman
st.set_page_config(
    page_title="Detektor Masker Wajah",
    page_icon="😷",
    layout="centered"
)

st.title("😷 Detektor Masker Wajah (CNN - Kamera & Upload)")
st.write("Aplikasi untuk mendeteksi penggunaan masker wajah secara real-time via Kamera atau Unggah Foto.")

# 2. Memuat Model .tflite Menggunakan OpenCV DNN
@st.cache_resource
def load_opencv_model():
    # OpenCV DNN membaca format .tflite secara native
    net = cv2.dnn.readNetFromTensorflow("model_masker.tflite")
    return net

try:
    net = load_opencv_model()
    st.success("Model .tflite Berhasil Dimuat!")
except Exception as e:
    st.error(f"Gagal memuat model 'model_masker.tflite'. Pastikan file ada di repo. Error: {e}")

# 3. Menu Pilihan Metode Input Gambar
Metode_input = st.selectbox(
    "Pilih Metode Input Gambar:",
    ("Gunakan Kamera (Webcam)", "Unggah File Foto")
)

# Inisialisasi variabel gambar
image = None

if metode_input == "Gunakan Kamera (Webcam)":
    # Mengaktifkan fitur kamera bawaan Streamlit
    img_file_buffer = st.camera_input("Posisikan wajah Anda di tengah kamera")
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)

else:
    # Fitur upload file konvensional
    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Gambar yang diunggah', use_column_width=True)

# 4. Proses Prediksi Jika Gambar Sudah Tersedia
if image is not None:
    st.write("Sedang menganalisis wajah...")
    
    # Preprocessing Gambar (Sesuai spesifikasi training: 120x120)
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    img_array = np.array(image)
    
    # Membuat blob untuk OpenCV DNN (Resize ke 120x120 & Normalisasi 1./255)
    blob = cv2.dnn.blobFromImage(
        img_array, 
        scalefactor=1.0/255.0, 
        size=(120, 120), 
        swapRB=False, 
        crop=False
    )
    
    # Jalankan Inferensi/Prediksi
    net.setInput(blob)
    preds = net.forward()
    prediction = preds[0][0]
    
    # 5. Klasifikasi Hasil (Aktivasi Sigmoid)
    st.subheader("Hasil Analisis:")
    if prediction < 0.5:
        confidence = (1 - prediction) * 100
        st.success(f"✅ **Menggunakan Masker** (Tingkat Keyakinan: {confidence:.2f}%)")
    else:
        confidence = prediction * 100
        st.error(f"❌ **Tidak Menggunakan Masker** (Tingkat Keyakinan: {confidence:.2f}%)")
        
    with st.expander("Lihat Detail Output Model"):
        st.write(f"Nilai mentah aktivasi Sigmoid: {prediction:.4f}")