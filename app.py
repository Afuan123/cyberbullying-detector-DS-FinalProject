"""
Cyberbullying Tweet Classification Dashboard
The Wizard Group — Final Project Data Science Batch 62

Pipeline: TF-IDF + Logistic Regression
"""

import streamlit as st
import joblib
import re
import html
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Cyberbullying Tweet Classifier",
    page_icon="logo_wizard.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# EFEK KURSOR (AURORA SEPERTI SENTER)
# =========================================================
# JavaScript ini diload di background untuk membaca koordinat kursor dan
# mengirimkannya ke CSS variables (--cursor-x dan --cursor-y).
components.html("""
<script>
    const parentDoc = window.parent.document;
    parentDoc.addEventListener('mousemove', (e) => {
        const x = (e.clientX / window.parent.innerWidth) * 100;
        const y = (e.clientY / window.parent.innerHeight) * 100;
        parentDoc.documentElement.style.setProperty('--cursor-x', x + '%');
        parentDoc.documentElement.style.setProperty('--cursor-y', y + '%');
    });
</script>
""", height=0, width=0)

# =========================================================
# CUSTOM CSS — THE WIZARD GROUP THEME
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;600;700&family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg-deep: #070a1c;
    --bg-panel: #10173a;
    --bg-panel-soft: #141d47;
    --navy-border: #232f63;
    --gold: #f2a93b;
    --gold-strong: #ffc465;
    --gold-soft: rgba(242, 169, 59, 0.35);
    --blue: #4f79f7;
    --blue-soft: rgba(79, 121, 247, 0.16);
    --text-primary: #f2f4fc;
    --text-muted: #96a0c7;
    
    /* Warna Notifikasi */
    --ruby: #ef5468;
    --ruby-soft: rgba(239, 84, 104, 0.14);
    --safe-green: #10b981; /* Hijau yang lebih seimbang, tidak terlalu neon */
    --safe-green-soft: rgba(16, 185, 129, 0.14);
    --warning-yellow: #f59e0b; /* Kuning/Amber yang elegan */
    --warning-yellow-soft: rgba(245, 158, 11, 0.14);
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background-color: var(--bg-deep);
    /* Latar belakang menggunakan kursor sebagai titik pusat senter */
    background-image:
        radial-gradient(circle 500px at var(--cursor-x, 50%) var(--cursor-y, 50%), rgba(79, 121, 247, 0.35) 0%, transparent 80%),
        radial-gradient(circle 400px at calc(100% - var(--cursor-x, 50%)) calc(100% - var(--cursor-y, 50%)), rgba(147, 179, 255, 0.15) 0%, transparent 80%),
        radial-gradient(1.6px 1.6px at 20% 28%, rgba(255,255,255,0.45) 0%, transparent 60%),
        radial-gradient(1.6px 1.6px at 72% 14%, rgba(255,255,255,0.3) 0%, transparent 60%),
        radial-gradient(1.4px 1.4px at 85% 55%, rgba(242,169,59,0.5) 0%, transparent 60%),
        radial-gradient(1.4px 1.4px at 38% 78%, rgba(255,255,255,0.25) 0%, transparent 60%);
    background-attachment: fixed;
    transition: background 0.15s ease-out; /* Memberikan efek halus saat mengikuti mouse */
}

/* =========================================================
   PENGATURAN CUSTOM: SEMBUNYIKAN DEPLOY, FOOTER, & MENU BAWAAN
   ========================================================= */
footer {visibility: hidden !important;} 
#MainMenu {visibility: hidden !important;} 
.viewerBadge_container__1QSob {display: none !important;}
.stDeployButton {display: none !important;} 

header[data-testid="stHeader"] {
    background: transparent !important;
    box-shadow: none !important;
}

section[data-testid="stSidebar"] {
    background-color: var(--bg-panel);
    border-right: 1px solid var(--navy-border);
}

h1, h2, h3, h4 { color: var(--text-primary); }

[data-testid="stMarkdownContainer"] h3 {
    font-family: 'Cinzel', serif;
    font-weight: 600;
    font-size: 1.2rem;
    letter-spacing: .2px;
    display: inline-block;
    padding-bottom: .4rem;
    border-bottom: 2px solid var(--gold-soft);
    margin-bottom: .7rem;
}

section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
    font-family: 'Cinzel', serif;
    font-weight: 600;
    font-size: 1.02rem;
    color: var(--text-primary);
    margin-top: .2rem;
}

[data-testid="stWidgetLabel"] p {
    color: var(--text-muted) !important;
    font-size: .85rem !important;
}

[data-testid="stHorizontalBlock"] { align-items: flex-start; }

/* hero */
.hero-title {
    font-family: 'Cinzel', serif;
    font-weight: 700;
    font-size: clamp(1.6rem, 3vw, 2.35rem);
    line-height: 1.15;
    margin: 0 0 .4rem 0;
    background: linear-gradient(90deg, #8fabff 0%, var(--gold) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-subtitle {
    font-size: .98rem;
    color: var(--text-muted);
    max-width: 72ch;
    line-height: 1.55;
    margin: 0;
}
.gold-rule {
    height: 1px;
    border: none;
    margin: 1.5rem 0;
    background: linear-gradient(90deg, transparent, var(--gold-soft) 20%, var(--gold) 50%, var(--gold-soft) 80%, transparent);
}

/* sidebar */
.sidebar-brand {
    font-family: 'Cinzel', serif;
    font-weight: 600;
    font-size: 1.15rem;
    text-align: center;
    background: linear-gradient(90deg, #8fabff 0%, var(--gold) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: .4rem 0 1.1rem 0;
}
.sidebar-card {
    background: var(--bg-panel-soft);
    border: 1px solid var(--navy-border);
    border-left: 3px solid var(--blue);
    border-radius: 10px;
    padding: 1rem 1.1rem;
    font-size: .89rem;
    line-height: 1.65;
    color: var(--text-muted);
}
.sidebar-card b { color: var(--text-primary); }
.sidebar-card ul { margin: .35rem 0 0 0; padding-left: 1.1rem; }
.tip-card {
    background: var(--bg-panel-soft);
    border: 1px solid var(--navy-border);
    border-left: 3px solid var(--gold);
    border-radius: 10px;
    padding: .85rem 1rem;
    font-size: .85rem;
    color: var(--text-muted);
    line-height: 1.55;
    margin-top: 1rem;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--bg-panel);
    border: 1px solid var(--navy-border) !important;
    border-radius: 18px !important;
    padding: .3rem .2rem;
}

/* text area */
.stTextArea textarea {
    background-color: var(--bg-panel-soft) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--navy-border) !important;
    border-radius: 12px !important;
    font-size: .95rem;
}
.stTextArea textarea:focus {
    border-color: var(--blue) !important;
    box-shadow: 0 0 0 3px var(--blue-soft) !important;
}
.stTextArea textarea::placeholder { color: #5c6690; }

/* buttons */
.stButton>button {
    border-radius: 10px;
    font-weight: 600;
    transition: transform .15s ease, box-shadow .15s ease, background .15s ease, border-color .15s ease;
}
button[kind="primary"] {
    background: var(--gold) !important;
    border: none !important;
    color: #0a0e27 !important;
    font-weight: 700 !important;
}
button[kind="primary"]:hover {
    background: var(--gold-strong) !important;
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(242, 169, 59, .35);
}
button[kind="secondary"] {
    background: linear-gradient(135deg, #7c3aed 0%, #3b82f6 100%) !important;
    border: 1px solid rgba(168, 85, 247, 0.4) !important;
    color: #ffffff !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: .85rem !important;
}
button[kind="secondary"]:hover {
    background: linear-gradient(135deg, #8b5cf6 0%, #2563eb 100%) !important;
    border-color: #f2a93b !important;
    color: #ffffff !important;
    box-shadow: 0 4px 14px rgba(124, 58, 237, 0.4);
    transform: translateY(-1px);
}

/* verdict card */
.verdict-card {
    display: flex;
    align-items: flex-start;
    gap: .9rem;
    padding: 1.1rem 1.3rem;
    border-radius: 14px;
    margin-bottom: 1rem;
}
.verdict-safe { background: var(--safe-green-soft); border: 1px solid rgba(16, 185, 129, 0.35); }
.verdict-alert { background: var(--ruby-soft); border: 1px solid rgba(239, 84, 104, 0.4); }
.verdict-warning { background: var(--warning-yellow-soft); border: 1px solid rgba(245, 158, 11, 0.4); }

.verdict-icon {
    flex-shrink: 0;
    width: 34px; height: 34px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
}
.verdict-safe .verdict-icon { background: var(--safe-green); color: #022c22; }
.verdict-alert .verdict-icon { background: var(--ruby); color: #2b0a0e; }
.verdict-warning .verdict-icon { background: var(--warning-yellow); color: #451a03; }

.verdict-title { font-weight: 700; font-size: 1.02rem; margin-bottom: .2rem; color: var(--text-primary); }
.verdict-desc { font-size: .86rem; color: var(--text-muted); line-height: 1.5; }

[data-testid="stPlotlyChart"] {
    background: radial-gradient(circle at 50% 35%, rgba(79,121,247,0.16), transparent 72%);
    border-radius: 18px;
}

/* empty state */
.empty-state {
    text-align: center;
    padding: 3rem 1.5rem;
    border: 1px dashed var(--navy-border);
    border-radius: 18px;
    background: var(--bg-panel-soft);
}
.empty-state-icon { font-size: 2.1rem; color: var(--gold); margin-bottom: .6rem; }
.empty-state-title { font-weight: 700; font-size: 1rem; color: var(--text-primary); margin-bottom: .4rem; }
.empty-state-desc { font-size: .87rem; color: var(--text-muted); max-width: 40ch; margin: 0 auto; line-height: 1.6; }

/* history */
.history-row {
    display: flex;
    align-items: center;
    gap: .7rem;
    padding: .55rem .2rem;
    border-bottom: 1px solid var(--navy-border);
    font-size: .85rem;
}
.history-row:last-child { border-bottom: none; }
.history-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.history-dot-safe { background: var(--safe-green); }
.history-dot-alert { background: var(--ruby); }
.history-dot-warning { background: var(--warning-yellow); }
.history-text { flex: 1; color: var(--text-primary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.history-tag { color: var(--text-muted); font-size: .78rem; }
.history-confidence { color: var(--gold); font-size: .78rem; font-weight: 600; min-width: 3rem; text-align: right; }

[data-testid="stExpander"] {
    background: var(--bg-panel-soft);
    border: 1px solid var(--navy-border) !important;
    border-radius: 12px !important;
}

/* footer */
.footer-text { text-align: center; color: var(--text-muted); font-size: .82rem; line-height: 1.7; }
.footer-brand { color: var(--gold); font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# STATE
# =========================================================
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "history" not in st.session_state:
    st.session_state.history = []

# =========================================================
# FUNGSI BANTUAN
# =========================================================
def clean_text(text):
    text = html.unescape(str(text))
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'\brt\b', ' ', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@st.cache_resource
def load_model():
    return joblib.load("cyberbullying_model.pkl")

try:
    model_pipeline = load_model()
except Exception as e:
    st.error(
        "Gagal memuat model. Pastikan file **cyberbullying_model.pkl** berada di folder "
        f"yang sama dengan aplikasi ini.\n\nDetail error: {e}"
    )
    st.stop()

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.image("logo_wizard.png", width="stretch")
    st.markdown('<div class="sidebar-brand">The Wizard Group</div>', unsafe_allow_html=True)

    st.header("Tentang proyek")
    st.markdown("""
        <div class="sidebar-card">
        Dashboard analitik untuk mendeteksi potensi <b>cyberbullying</b> pada teks menggunakan machine learning.
        <ul>
            <li>Tim: The Wizard Group</li>
            <li>Batch: Data Science Batch 62</li>
            <li>Model: TF-IDF + Logistic Regression</li>
            <li>Bahasa teks: Inggris</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="tip-card">
    Coba kalimat formal, gaul, maupun netral untuk melihat sejauh mana model dapat membedakannya.
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# HERO HEADER
# =========================================================
st.markdown('<div class="hero-title">Cyberbullying Tweet Classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">Dashboard analitik oleh The Wizard Group untuk mendeteksi indikasi '
    'cyberbullying pada teks berbahasa Inggris secara real-time, lengkap dengan tingkat keyakinan model.</p>',
    unsafe_allow_html=True,
)
st.markdown('<hr class="gold-rule">', unsafe_allow_html=True)

# =========================================================
# LAYOUT UTAMA
# =========================================================
col_input, col_result = st.columns([1, 1.2], gap="large")

with col_input:
    st.subheader("Masukkan teks")

    with st.container(border=True):
        ex_col1, ex_col2 = st.columns(2)
        if ex_col1.button("Contoh kalimat aman", width="stretch"):
            st.session_state.input_text = "The train schedule has been updated for tomorrow morning."
        if ex_col2.button("Contoh kalimat berisiko", width="stretch"):
            st.session_state.input_text = "You are so stupid, nobody wants you, go away !"

        user_input = st.text_area(
            "Ketik atau tempel tweet berbahasa Inggris di sini",
            key="input_text",
            height=160,
            placeholder="Contoh: You are amazing and I love your work...",
        )

        predict_button = st.button("Analisis Teks", type="primary", width="stretch")

with col_result:
    st.subheader("Hasil evaluasi model")

    if predict_button:
        if not user_input.strip():
            st.warning("Mohon masukkan teks terlebih dahulu sebelum menganalisis.")
        else:
            with st.spinner("Menganalisis teks melalui model..."):
                cleaned_input = clean_text(user_input)
                prediction = model_pipeline.predict([cleaned_input])[0]

                # Mendapatkan probability untuk setiap kelas
                probabilities = np.atleast_1d(
                    model_pipeline.predict_proba([cleaned_input])[0]
                ).astype(float)

                # Urutan kelas eksplisit sesuai request
                desired_order = [
                    "other_cyberbullying",
                    "not_cyberbullying",
                    "religion",
                    "gender",
                    "ethnicity",
                    "age"
                ]

                # Mapping probabilitas asli berdasarkan class agar urutan bisa diubah
                prob_dict = dict(zip(model_pipeline.classes_, probabilities))

                ordered_classes = [c for c in desired_order if c in prob_dict]
                # Menambahkan kelas sisa jika ada yang tidak terdaftar di desired_order (safety check)
                for c in prob_dict:
                    if c not in ordered_classes:
                        ordered_classes.append(c)

                ordered_probs = [prob_dict[c] for c in ordered_classes]

                score_df = pd.DataFrame({
                    "Kategori": ordered_classes,
                    "Skor": ordered_probs,
                })

                top_confidence_pct = float(np.max(probabilities) * 100)

                st.session_state.history.insert(0, {
                    "text": user_input.strip(),
                    "prediction": str(prediction),
                    "confidence": top_confidence_pct,
                })
                st.session_state.history = st.session_state.history[:8]

            # ---- KARTU VERDICT (NOTIFIKASI) ----
            if prediction == "not_cyberbullying":
                # Hijau (Aman)
                icon_svg = (
                    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" '
                    'stroke="currentColor" stroke-width="2.4" stroke-linecap="round" '
                    'stroke-linejoin="round"><circle cx="12" cy="12" r="9"></circle>'
                    '<polyline points="8 12.5 11 15.5 16 9"></polyline></svg>'
                )
                st.markdown(f"""
                <div class="verdict-card verdict-safe">
                    <div class="verdict-icon">{icon_svg}</div>
                    <div>
                        <div class="verdict-title">Aman (Not Cyberbullying)</div>
                        <div class="verdict-desc">Teks ini menunjukkan pola bahasa percakapan normal tanpa indikasi konten berbahaya.</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            elif prediction == "other_cyberbullying":
                # Merah (Alert)
                icon_svg = (
                    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" '
                    'stroke="currentColor" stroke-width="2.4" stroke-linecap="round" '
                    'stroke-linejoin="round"><path d="M12 3 L22 20 L2 20 Z"></path>'
                    '<line x1="12" y1="9" x2="12" y2="14"></line>'
                    '<circle cx="12" cy="17" r="0.6" fill="currentColor"></circle></svg>'
                )
                label = html.escape(str(prediction).replace("_", " ").title())
                st.markdown(f"""
                <div class="verdict-card verdict-alert">
                    <div class="verdict-icon">{icon_svg}</div>
                    <div>
                        <div class="verdict-title">Terindikasi {label}</div>
                        <div class="verdict-desc">Pola bahasa pada teks ini masuk dalam klasifikasi bentuk cyberbullying lainnya.</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            else:
                # Kuning (Warning) untuk kategori: Religion, Gender, Ethnicity, Age
                icon_svg = (
                    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" '
                    'stroke="currentColor" stroke-width="2.4" stroke-linecap="round" '
                    'stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>'
                    '<line x1="12" y1="9" x2="12" y2="13"></line>'
                    '<line x1="12" y1="17" x2="12.01" y2="17"></line></svg>'
                )
                label = html.escape(str(prediction).replace("_", " ").title())
                st.markdown(f"""
                <div class="verdict-card verdict-warning">
                    <div class="verdict-icon">{icon_svg}</div>
                    <div>
                        <div class="verdict-title">Terindikasi {label} Cyberbullying</div>
                        <div class="verdict-desc">Pola bahasa pada teks ini cocok dengan karakteristik cyberbullying untuk kategori terkait.</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ---- GAUGE METER ----
            if prediction == "not_cyberbullying":
                gauge_color = "#10b981" # Hijau
            elif prediction == "other_cyberbullying":
                gauge_color = "#ef5468" # Merah
            else:
                gauge_color = "#f59e0b" # Kuning

            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=top_confidence_pct,
                number={"suffix": "%", "font": {"size": 38, "color": "#f2f4fc"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#5c6690", "tickfont": {"color": "#8b93b8", "size": 10}},
                    "bar": {"color": gauge_color, "thickness": 0.28},
                    "bgcolor": "rgba(255,255,255,0.03)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 100], "color": "rgba(255,255,255,0.04)"},
                    ],
                },
            ))
            gauge_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#f2f4fc", "family": "Inter"},
                height=200,
                margin=dict(l=20, r=20, t=20, b=10),
            )
            st.plotly_chart(gauge_fig, config={"displayModeBar": False})
            st.caption("Skor confidence dari predict_proba model.")

            # ---- GRAFIK BAR (Kategori spesifik berdasar warna) ----
            bar_colors = []
            for cat in score_df["Kategori"]:
                if cat == "not_cyberbullying":
                    bar_colors.append("#10b981") # Hijau
                elif cat == "other_cyberbullying":
                    bar_colors.append("#ef5468") # Merah
                else:
                    bar_colors.append("#f59e0b") # Kuning (religion, gender, ethnicity, age)

            fig = px.bar(score_df, x="Skor", y="Kategori", orientation="h", text_auto=".2f")
            fig.update_traces(marker_color=bar_colors, marker_line_width=0, textposition="outside", textfont_color="#f2f4fc")
            fig.update_layout(
                title=dict(text="Confidence score tiap kategori", font=dict(family="Inter", size=14, color="#f2f4fc")),
                yaxis={
                    "categoryorder": "array",
                    "categoryarray": ordered_classes[::-1], # Dibalik agar urutannya dari atas ke bawah pada bar chart sesuai array
                    "title": None,
                    "color": "#c7cdea",
                },
                xaxis={"title": None, "color": "#c7cdea", "gridcolor": "rgba(255,255,255,0.06)"},
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_family="Inter",
                font_color="#f2f4fc",
                margin=dict(l=10, r=10, t=45, b=10),
                height=280,
                showlegend=False,
            )
            st.plotly_chart(fig, config={"displayModeBar": False})

    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">&#9680;</div>
            <div class="empty-state-title">Belum ada teks yang dianalisis</div>
            <div class="empty-state-desc">Tulis atau tempel teks di panel sebelah kiri, lalu jalankan tombol Analisis Teks untuk melihat status, tingkat keyakinan, dan rincian skor tiap kategori.</div>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# RIWAYAT ANALISIS SESI
# =========================================================
if st.session_state.history:
    st.markdown('<hr class="gold-rule">', unsafe_allow_html=True)
    with st.expander(f"Riwayat analisis sesi ini ({len(st.session_state.history)})"):
        for item in st.session_state.history:
            raw_text = item["text"]
            snippet = raw_text[:80] + ("…" if len(raw_text) > 80 else "")
            snippet = html.escape(snippet)
            pred = item["prediction"]

            if pred == "not_cyberbullying":
                dot_class = "history-dot-safe"
                label = "Aman"
            elif pred == "other_cyberbullying":
                dot_class = "history-dot-alert"
                label = "Other Cyberbullying"
            else:
                dot_class = "history-dot-warning"
                label = html.escape(pred.replace("_", " ").title())

            st.markdown(f"""
            <div class="history-row">
                <span class="history-dot {dot_class}"></span>
                <span class="history-text">{snippet}</span>
                <span class="history-tag">{label}</span>
                <span class="history-confidence">{item['confidence']:.0f}%</span>
            </div>
            """, unsafe_allow_html=True)

        if st.button("Bersihkan riwayat", key="clear_history"):
            st.session_state.history = []
            st.rerun()

# =========================================================
# FOOTER
# =========================================================
st.markdown('<hr class="gold-rule">', unsafe_allow_html=True)
st.markdown("""
<p class="footer-text">
Dibangun oleh <span class="footer-brand">The Wizard Group</span><br>
Final Project Data Science Batch 62
</p>
""", unsafe_allow_html=True)