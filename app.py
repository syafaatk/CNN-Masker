import streamlit as st
import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image

# 1. Konfigurasi Halaman
st.set_page_config(
    page_title="Detektor Masker Wajah",
    page_icon="😷",
    layout="centered"
)

st.title("😷 Detektor Masker Wajah (CNN - TFLite)")
st.write("Unggah foto wajah Anda untuk mendeteksi penggunaan masker.")

# 2. Load Model TFLite (Menggunakan cache agar efisien)
@st.cache_resource
def load_tflite_model():
    interpreter = tflite.Interpreter(model_path="model_masker.tflite")
    interpreter.allocate_tensors()
    return interpreter

try:
    interpreter = load_tflite_model()
    st.success("Model TFLite Ringan Berhasil Dimuat!")
    
    # Ambil informasi detail input & output tensor
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    st.error(f"Gagal memuat model 'model_masker.tflite'. Error: {e}")

# 3. Fitur Upload Gambar
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar yang diunggah', use_column_width=True)
    
    st.write("Sedang menganalisis...")
    
    # 4. Preprocessing Gambar (120x120)
    IMG_SIZE = (120, 120)
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    img_resized = image.resize(IMG_SIZE)
    img_array = np.array(img_resized, dtype=np.float32)
    
    # Normalisasi skala 1./255
    img_array = img_array / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    
    # 5. Jalankan Inferensi TFLite
    interpreter.set_tensor(input_details[0]['index'], img_batch)
    interpreter.invoke()
    
    # Mengambil hasil prediksi mentah (output sigmoid)
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
    
    # 6. Klasifikasi Hasil
    st.subheader("Hasil Analisis:")
    if prediction < 0.5:
        confidence = (1 - prediction) * 100
        st.success(f"✅ **Menggunakan Masker** (Tingkat Keyakinan: {confidence:.2f}%)")
    else:
        confidence = prediction * 100
        st.error(f"❌ **Tidak Menggunakan Masker** (Tingkat Keyakinan: {confidence:.2f}%)")
        
    with st.expander("Lihat Detail Output Model"):
        st.write(f"Nilai mentah aktivasi Sigmoid: {prediction:.4f}")