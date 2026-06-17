import streamlit as st
import numpy as np
from PIL import Image

# 1. Konfigurasi Halaman
st.set_page_config(
    page_title="Detektor Masker Wajah",
    page_icon="😷",
    layout="centered"
)

st.title("😷 Detektor Masker Wajah (CNN - Lightweight Engine)")
st.write("Aplikasi deteksi penggunaan masker wajah via Kamera atau Unggah Foto.")

# 2. Memuat Interpreter TFLite Menggunakan Library Bawaan Python (Sangat Aman)
@st.cache_resource
def load_tflite_interpreter():
    try:
        # Mencoba memuat menggunakan library standar streamlit cloud
        import tflite_runtime.interpreter as tflite
        interpreter = tflite.Interpreter(model_path="model_masker.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except ImportError:
        try:
            # Fallback jika berjalan di environment lokal dengan tensorflow penuh
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(model_path="model_masker.tflite")
            interpreter.allocate_tensors()
            return interpreter
        except Exception as e:
            st.error(f"Gagal memuat engine model. Error: {e}")
            return None

interpreter = load_tflite_interpreter()

if interpreter is not None:
    st.success("Model .tflite Berhasil Dimuat!")
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
else:
    st.error("⚠️ Aplikasi dihentikan karena model gagal dimuat.")
    st.stop()

# 3. Menu Pilihan Metode Input Gambar
metode_input = st.selectbox(
    "Pilih Metode Input Gambar:",
    ("Gunakan Kamera (Webcam)", "Unggah File Foto")
)

image = None

if metode_input == "Gunakan Kamera (Webcam)":
    img_file_buffer = st.camera_input("Posisikan wajah Anda di tengah kamera")
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
else:
    uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Gambar yang diunggah', use_column_width=True)

# 4. Proses Prediksi Menggunakan NumPy & Interpreter TFLite
if image is not None:
    st.write("Sedang menganalisis wajah...")
    
    # Preprocessing Gambar (Resize ke 120x120 dan konversi ke RGB)
    IMG_SIZE = (120, 120)
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    img_resized = image.resize(IMG_SIZE)
    img_array = np.array(img_resized, dtype=np.float32)
    
    # Normalisasi (Skala 1./255) sesuai proses training Anda
    img_array = img_array / 255.0
    
    # Tambahkan dimensi batch -> (1, 120, 120, 3)
    img_batch = np.expand_dims(img_array, axis=0)
    
    try:
        # Jalankan Inferensi TFLite secara manual (Sangat ringan & cepat)
        interpreter.set_tensor(input_details[0]['index'], img_batch)
        interpreter.invoke()
        
        # Mengambil hasil akhir dari layer output (Aktivasi Sigmoid)
        prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
        
        # 5. Menampilkan Hasil Klasifikasi
        st.subheader("Hasil Analisis:")
        if prediction < 0.5:
            confidence = (1 - prediction) * 100
            st.success(f"✅ **Menggunakan Masker** (Tingkat Keyakinan: {confidence:.2f}%)")
        else:
            confidence = prediction * 100
            st.error(f"❌ **Tidak Menggunakan Masker** (Tingkat Keyakinan: {confidence:.2f}%)")
            
        with st.expander("Lihat Detail Output Model"):
            st.write(f"Nilai mentah aktivasi Sigmoid: {prediction:.4f}")
            
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")