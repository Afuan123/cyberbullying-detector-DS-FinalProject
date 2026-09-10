import streamlit as st
import joblib
import re
import html
import pandas as pd
import plotly.express as px

# Konfigurasi halaman agar tampil secara profesional (Wide Mode)
st.set_page_config(
    page_title="Cyberbullying Tweet Classifier",
    page_icon="🤖",
    layout="wide"
)


# Fungsi pembersih teks (harus konsisten dengan training)
def clean_text(text):
    text = html.unescape(str(text))
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'\brt\b', ' ', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Load Pipeline Model
@st.cache_resource
def load_model():
    return joblib.load('cyberbullying_model.pkl')


try:
    model_pipeline = load_model()
except Exception as e:
    st.error(f"Gagal memuat model. Pastikan file 'cyberbullying_model.pkl' ada di satu folder yang sama. Error: {e}")
    st.stop()

# --- HEADER APLIKASI ---
st.title("🤖 Cyberbullying Tweet Classification Dashboard")
st.markdown("""
Aplikasi web analitik ini dirancang untuk mendeteksi potensi *cyberbullying* pada teks *tweet* menggunakan Machine Learning Pipeline (**TF-IDF + SGD Classifier**). 
Masukkan teks pada panel di bawah untuk melihat hasil klasifikasi secara real-time beserta tingkat keyakinan model (*confidence scores*).
""")

st.markdown("---")

# Layout Kolom: Kiri untuk Input, Kanan untuk Hasil Analisis
col_input, col_result = st.columns([1, 1.2], gap="large")

with col_input:
    st.subheader("✍️ Masukkan Input Teks")
    user_input = st.text_area(
        "Ketik atau tempel tweet bahasa Inggris di sini:",
        height=180,
        placeholder="Contoh: You are amazing and I love your work..."
    )

    predict_button = st.button("🚀 Analisis Teks", type="primary", use_container_width=True)

with col_result:
    st.subheader("📊 Hasil Evaluasi Model")

    if predict_button:
        if not user_input.strip():
            st.warning("⚠️ Mohon masukkan teks terlebih dahulu sebelum menganalisis!")
        else:
            with st.spinner("Sedang memproses teks melalui model..."):
                # 1. Bersihkan teks
                cleaned_input = clean_text(user_input)

                # 2. Prediksi Kategori
                prediction = model_pipeline.predict([cleaned_input])[0]

                # 3. Ambil decision function untuk skor tiap kelas
                confidence_scores = model_pipeline.decision_function([cleaned_input])[0]
                classes = model_pipeline.classes_

                # Buat DataFrame dan urutkan dari skor tertinggi ke terendah
                score_df = pd.DataFrame({
                    'Kategori': classes,
                    'Skor': confidence_scores
                }).sort_values(by='Skor', ascending=False).reset_index(drop=True)

            # --- TAMPILAN HASIL UTAMA (METRIC Keren) ---
            if prediction == "not_cyberbullying":
                st.success(f"### ✨ Prediksi: AMAN (Not Cyberbullying)")
            else:
                st.error(f"### ⚠️ Prediksi Terdeteksi: {prediction.upper()}")

            st.markdown(f"**Teks yang dibersihkan:** `{cleaned_input}`")

            # --- GRAFIK PLOTLY HORIZONTAL YANG RAPI ---
            # Grafik horizontal membuat nama kategori panjang terbaca dengan sangat jelas
            fig = px.bar(
                score_df,
                x='Skor',
                y='Kategori',
                orientation='h',
                title='Peringkat Confidence Score tiap Kategori',
                text_auto='.2f',
                color='Skor',
                color_continuous_scale='Blues'
            )

            # Membalik sumbu y agar kategori dengan skor tertinggi berada di paling atas
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                margin=dict(l=20, r=20, t=40, b=20),
                height=350
            )

            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "👈 Masukkan teks pada kolom di sebelah kiri, lalu klik tombol **'Analisis Teks'** untuk melihat hasil prediksi.")

# Footer Konsultasi Gaya Profesional
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Final Project Data Science Batch 62 • Group 3 Cyberbullying Classification</p>",
    unsafe_allow_html=True)