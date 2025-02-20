import io
import json
import numpy as np
import tensorflow as tf
import streamlit as st
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load daftar kelas dari file JSON
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Konversi indeks ke nama kelas untuk dropdown
class_labels = list(class_indices.keys())

# Load model
model = load_model("signature_model_mobilenetv2.h5", custom_objects={'KerasLayer': tf.keras.layers.Layer})

### PERBAIKAN GAMBAR UNTUK KAMERA BURUK ###
def enhance_brightness_contrast(image, alpha=1.5, beta=50):
    """Meningkatkan kecerahan dan kontras gambar"""
    enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return enhanced

def remove_noise(image):
    """Menghilangkan noise dari gambar"""
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    return denoised

def sharpen_image(image):
    """Meningkatkan ketajaman gambar"""
    gaussian_blur = cv2.GaussianBlur(image, (0, 0), 3)
    sharpened = cv2.addWeighted(image, 1.5, gaussian_blur, -0.5, 0)
    return sharpened

def preprocess_signature(image):
    """Preprocessing gambar sebelum dikirim ke model"""
    # Konversi ke grayscale
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Resize gambar
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)

    # Thresholding untuk memperjelas tanda tangan
    _, image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Pastikan tanda tangan tetap hitam dan background putih
    if np.mean(image) < 127:
        image = cv2.bitwise_not(image)

    # Ekstraksi kontur tanda tangan
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        image = image[y:y+h, x:x+w]  # Crop ke area tanda tangan

    # Konversi ke 3-channel (RGB)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Perbaiki kualitas gambar sebelum masuk ke model
    image = enhance_brightness_contrast(image)
    image = remove_noise(image)
    image = sharpen_image(image)

    # Resize lagi ke input model (224x224)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Normalisasi
    image = image.astype(np.float32) / 255.0

    return image

def draw_similarity_box(image, similarity_score):
    """Menambahkan kotak sebagai indikator prosentase kemiripan pada gambar"""
    # Konversi gambar ke OpenCV format
    image = np.array(image)

    # Tentukan warna berdasarkan skor kemiripan
    if similarity_score > 75:
        color = (0, 255, 0)  # Hijau (Sangat Mirip)
    elif similarity_score > 50:
        color = (255, 255, 0)  # Kuning (Cukup Mirip)
    else:
        color = (255, 0, 0)  # Merah (Kurang Mirip)

    # Tambahkan kotak sebagai indikator kemiripan
    height, width, _ = image.shape
    cv2.rectangle(image, (10, 10), (width - 10, height - 10), color, 5)

    # Tambahkan teks persen di atas gambar
    text = f"Similarity: {similarity_score:.2f}%"
    cv2.putText(image, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    return image

def import_and_predict(image_data, model, selected_label):
    """Melakukan prediksi dan menampilkan kemiripan (%) dengan Nasabah yang dipilih"""

    # Load gambar ke format OpenCV
    image = Image.open(image_data).convert("RGB")
    image = np.array(image)

    # Preprocessing gambar sebelum masuk ke model
    preprocessed_image = preprocess_signature(image)
    
    # Konversi ke array TensorFlow (tambahkan batch dimension)
    img_array = np.expand_dims(preprocessed_image, axis=0)

    # Pastikan shape input sesuai dengan model (1, 224, 224, 3)
    st.write(f"Input Shape ke Model: {img_array.shape}")

    # Prediksi dengan model
    predictions = model.predict(img_array)

    # Cari indeks dari kelas yang dipilih di dropdown
    selected_idx = class_indices[selected_label]  

    # Ambil probabilitas kelas yang dipilih
    similarity_score = predictions[0][selected_idx] * 100  # Konversi ke persentase

    # Gambar indikator kemiripan
    result_image = draw_similarity_box(image, similarity_score)

    return result_image, f"Kemiripan dengan {selected_label}: {similarity_score:.2f}%"

def run():
    st.title("Signature Verification with Camera Input (Enhanced)")

    # Dropdown untuk memilih kelas tanda tangan
    selected_label = st.selectbox("Pilih Nama Nasabah:", class_labels)

    # Checkbox untuk mengaktifkan kamera
    enable = st.checkbox("Enable Camera")

    # Ambil gambar dari kamera
    picture = st.camera_input("Take a picture", disabled=not enable)

    # Atau unggah gambar dari file
    file = st.file_uploader("Upload tanda tangan untuk verifikasi", type=["jpg", "png"])

    # Gunakan gambar dari kamera jika tersedia, jika tidak gunakan file upload
    if picture:
        image_data = io.BytesIO(picture.getvalue())  # Konversi gambar ke format yang bisa diproses
        st.image(picture, caption="Gambar dari Kamera", use_container_width=True)
        result_image, result_text = import_and_predict(image_data, model, selected_label)
        st.image(result_image, caption="Hasil Prediksi dengan Indikator", use_container_width=True)
        st.write(result_text)

    elif file:
        st.image(file, caption="Gambar yang diunggah", use_container_width=True)
        result_image, result_text = import_and_predict(file, model, selected_label)
        st.image(result_image, caption="Hasil Prediksi dengan Indikator", use_container_width=True)
        st.write(result_text)

if __name__ == "__main__":
    run()