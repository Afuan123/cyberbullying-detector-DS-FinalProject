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

# --- CUSTOM CSS STYLING (The Wizard Group Theme) ---
st.markdown("""
    <style>
    /* Mengatur latar belakang utama aplikasi */
    .stApp {
        background-color: #0b0f19;
        background-image: radial-gradient(circle at 10% 20%, rgba(124, 58, 237, 0.08) 0%, transparent 40%),
                          radial-gradient(circle at 90% 80%, rgba(59, 130, 246, 0.08) 0%, transparent 40%);
    }

    /* Mempercantik kotak kontainer / kartu */
    div[data-testid="stVerticalBlock"] > div:has(div.stMarkdown) {
        border-radius: 12px;
    }

    /* Tombol utama dengan aksen gradasi ungu/biru modern */
    .stButton>button {
        border-radius: 8px;
        font-weight: bold;
        background: linear-gradient(135deg, #7c3aed 0%, #3b82f6 100%);
        color: white;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(124, 58, 237, 0.4);
    }

    /* Judul header dengan efek gradasi teks */
    .wizard-title {
        background: linear-gradient(93deg, #a855f7 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR INFORMASI (The Wizard Group) ---
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>🧙‍♂️✨</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #a855f7;'>The Wizard Group</h3>", unsafe_allow_html=True)
    st.markdown("---")

    st.header("Tentang Proyek")
    st.markdown("""
    **Cyberbullying Detection Dashboard**  
    Aplikasi web cerdas untuk mendeteksi potensi toksisitas dan *cyberbullying* pada teks menggunakan Machine Learning.

    * **Team:** The Wizard Group
    * **Batch:** Data Science Batch 62
    * **Model:** TF-IDF + SGD Classifier
    * **Bahasa:** Inggris (*English*)
    """)
    st.markdown("---")
    st.info(
        "💡 **Tips:** Uji performa model menggunakan kalimat formal, gaul, maupun kalimat netral untuk melihat batas analisisnya.")

# Inisialisasi session_state untuk teks input agar tidak hilang saat tombol diklik
if 'input_text' not in st.session_state:
    st.session_state.input_text = ""


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

    # Tombol Contoh Cepat (Quick Fill) menggunakan session_state
    st.markdown("Coba contoh kalimat instan:")
    ex_col1, ex_col2 = st.columns(2)

    if ex_col1.button("✨ Contoh Kalimat Aman"):
        st.session_state.input_text = "The train schedule has been updated for tomorrow morning."
    if ex_col2.button("⚠️ Contoh Cyberbullying"):
        st.session_state.input_text = "You are so stupid, nobody wants you here!"

    # Text area dihubungkan langsung dengan st.session_state.input_text
    user_input = st.text_area(
        "Ketik atau tempel tweet bahasa Inggris di sini:",
        key='input_text',
        height=160,
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
                cleaned_input = clean_text(user_input)
                prediction = model_pipeline.predict([cleaned_input])[0]
                confidence_scores = model_pipeline.decision_function([cleaned_input])[0]
                classes = model_pipeline.classes_

                score_df = pd.DataFrame({
                    'Kategori': classes,
                    'Skor': confidence_scores
                }).sort_values(by='Skor', ascending=False).reset_index(drop=True)

            # --- KARTU INDIKATOR UTAMA ---
            if prediction == "not_cyberbullying":
                st.success("### ✨ Status: AMAN (Not Cyberbullying)")
                st.markdown("Model mendeteksi teks ini sebagai percakapan normal/positif.")
            else:
                st.error(f"### ⚠️ Status: TERDETEKSI {prediction.upper()}")
                st.markdown("Teks ini memiliki indikasi pola bahasa yang masuk dalam kategori *cyberbullying*.")

            # Tampilkan teks yang sudah dibersihkan dalam bentuk expander
            with st.expander("🔍 Lihat Detail Preprocessing Teks"):
                st.write(f"**Teks Asli:** {user_input}")
                st.write(f"**Teks Setelah Dibersihkan (Cleaned):** `{cleaned_input}`")

            # --- GRAFIK PLOTLY HORIZONTAL ---
            fig = px.bar(
                score_df,
                x='Skor',
                y='Kategori',
                orientation='h',
                title='Peringkat Confidence Score tiap Kategori',
                text_auto='.2f',
                color='Skor',
                color_continuous_scale='Purples'
            )

            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                margin=dict(l=20, r=20, t=40, b=20),
                height=320
            )

            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👈 Masukkan teks atau pilih tombol contoh di sebelah kiri, lalu klik **'Analisis Teks'**.")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Final Project Data Science Batch 62 • The Wizard Group</p>",
    unsafe_allow_html=True
)