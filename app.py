import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Konfigurasi Halaman
st.set_page_config(
    page_title="Detektor Masker Wajah",
    page_icon="😷",
    layout="centered"
)

# 2. Judul Aplikasi
st.title("😷 Detektor Masker Wajah (CNN)")
st.write("Unggah foto wajah Anda untuk mendeteksi apakah Anda mengenakan masker atau tidak.")

# 3. Load Model (Menggunakan cache agar tidak dipanggil berulang kali)
@st.cache_resource
def load_my_model():
    # Pastikan nama file sesuai dengan file .h5 Anda di repo
    model = tf.keras.models.load_model('model_masker_terbaik.h5')
    return model

try:
    model = load_my_model()
    st.success("Model CNN Berhasil Dimuat!")
except Exception as e:
    st.error(f"Gagal memuat model. Pastikan 'model_masker_terbaik.h5' ada di repository. Error: {e}")

# 4. Fitur Upload Gambar
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar yang diunggah
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diunggah', use_column_width=True)
    
    st.write("Sedang menganalisis...")
    
    # 5. Preprocessing Gambar (Sesuaikan dengan spesifikasi training Anda: 120x120)
    IMG_SIZE = (120, 120)
    
    # Konversi ke RGB jika gambar punya alpha channel (RGBA)
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    img_resized = image.resize(IMG_SIZE)
    img_array = np.array(img_resized)
    
    # Normalisasi (1./255) sesuai dengan normalization_layer di notebook Anda
    img_array = img_array / 255.0
    
    # Expand dimensi menjadi (1, 120, 120, 3) karena model menerima batch
    img_batch = np.expand_dims(img_array, axis=0)
    
    # 6. Prediksi
    prediction = model.predict(img_batch)[0][0]
    
    # Logika klasifikasi binary_crossentropy (Sigmoid)
    # Catatan: Sesuaikan indeks label (0 atau 1) dengan urutan folder saat training.
    # Biasanya secara default alfabetis: 0 = dengan_masker, 1 = tanpa_masker
    # Jika terbalik pada model Anda, silakan tukar kondisinya di bawah ini.
    
    st.subheader("Hasil Analisis:")
    if prediction < 0.5:
        confidence = (1 - prediction) * 100
        st.success(f"✅ **Menggunakan Masker** (Tingkat Keyakinan: {confidence:.2f}%)")
    else:
        confidence = prediction * 100
        st.error(f"❌ **Tidak Menggunakan Masker** (Tingkat Keyakinan: {confidence:.2f}%)")
        
    # Opsional: Menampilkan nilai mentah output sigmoid
    with st.expander("Lihat Detail Output Model"):
        st.write(f"Nilai mentah aktivasi Sigmoid: {prediction:.4f}")