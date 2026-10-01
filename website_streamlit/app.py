"""
URL Sentinel — Deteksi Phishing URL berbasis Random Forest + Explainable AI (SHAP)
Aplikasi web lokal menggunakan Streamlit.

Jalankan dengan:
    streamlit run app.py
"""
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from features import (FEATURE_META, FEATURE_ORDER, extract_features_14,
                      get_hostname)

RANDOM_STATE = 42
DATASET_KAGGLE_PATH = "dataset_kaggle.csv"
DATASET_PHISHTANK_PATH = "dataset_phishtank.csv"
GLOBAL_SHAP_SAMPLE_SIZE = 500

# ----------------------------------------------------------------------------
# Konfigurasi halaman
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="URL Sentinel — Deteksi Phishing",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "model/random_forest_phishing_model.joblib"
COLS_PATH = "model/feature_columns.joblib"


# ----------------------------------------------------------------------------
# Load model (di-cache agar tidak dimuat ulang setiap interaksi)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model Random Forest ...")
def load_model():
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(COLS_PATH)
    explainer = shap.TreeExplainer(model)
    return model, feature_cols, explainer


model, feature_cols, explainer = load_model()


def get_base_value():
    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        return float(ev[1]) if len(ev) > 1 else float(ev[0])
    return float(ev)


# ----------------------------------------------------------------------------
# Insight Global (SHAP) — mereproduksi Bagian 6.1 notebook penelitian:
# dataset dibersihkan dengan proses yang identik (Sub-bab 4.1), lalu diambil sampel untuk
# dihitung nilai SHAP-nya sehingga bisa ditampilkan Summary Plot (global) dan
# Bar Plot gabungan (rata-rata |SHAP value| per fitur) di aplikasi web.
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_clean_dataset():
    """Menggabungkan & membersihkan dataset Kaggle + PhishTank.

    Langkahnya identik dengan notebook penelitian (Bagian 2) dan Sub-bab 4.1 laporan:
    concat -> hapus duplikat berdasarkan kolom url -> petakan label (legitimate=0,
    phishing=1) -> buang kategori lain (defacement/malware). Tidak ada perubahan huruf
    besar/kecil pada URL, karena notebook juga tidak melakukannya.
    """
    kaggle = pd.read_csv(DATASET_KAGGLE_PATH)
    phishtank = pd.read_csv(DATASET_PHISHTANK_PATH)

    kaggle["source"] = "Kaggle"
    phishtank["source"] = "PhishTank"

    dataset = pd.concat([kaggle, phishtank], ignore_index=True)
    dataset = dataset.drop_duplicates(subset="url")

    dataset["label"] = dataset["type"].map({"legitimate": 0, "phishing": 1})
    dataset = dataset.dropna(subset=["label"]).reset_index(drop=True)
    dataset["label"] = dataset["label"].astype(int)
    return dataset


@st.cache_data(show_spinner="Menyiapkan sampel data & menghitung nilai SHAP global ...")
def compute_global_shap(_model, _explainer, sample_size=GLOBAL_SHAP_SAMPLE_SIZE):
    """Mengambil sampel acak dari dataset, mengekstrak fitur, dan menghitung SHAP-nya.

    Parameter diawali underscore (_model, _explainer) agar Streamlit tidak
    mencoba melakukan hashing pada objek tersebut saat caching.
    """
    dataset = load_clean_dataset()
    n = min(sample_size, len(dataset))
    sample = dataset.sample(n=n, random_state=RANDOM_STATE).reset_index(drop=True)

    feature_rows = sample["url"].apply(lambda u: pd.Series(extract_features_14(u)))
    X_sample = feature_rows[FEATURE_ORDER]

    sv = _explainer.shap_values(X_sample)
    shap_values_phishing = sv[1] if isinstance(sv, list) else sv[:, :, 1]

    return X_sample, shap_values_phishing


def render_global_summary_plot(X_sample, shap_values_phishing):
    """SHAP Summary Plot (dot) — ringkasan global kontribusi & arah tiap fitur."""
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.summary_plot(
        shap_values_phishing, X_sample, plot_type="dot", show=False, plot_size=None
    )
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)


def render_global_bar_plot(feature_cols, shap_values_phishing):
    """Bar plot gabungan: rata-rata |SHAP value| per fitur (kepentingan global fitur)."""
    mean_abs_shap = pd.Series(
        np.abs(shap_values_phishing).mean(axis=0), index=feature_cols
    ).sort_values(ascending=True)

    labels = [f"{FEATURE_META[f]['no']}. {FEATURE_META[f]['label']}" for f in mean_abs_shap.index]
    values = mean_abs_shap.values

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(
                color=values,
                colorscale="Blues",
                line=dict(color="#1D4E89", width=0.5),
            ),
            text=[f"{v:.4f}" for v in values],
            textposition="outside",
            hovertemplate="%{y}<br>Rata-rata |SHAP value|: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=440,
        margin=dict(l=10, r=40, t=10, b=10),
        xaxis_title="Rata-rata |SHAP value| (semakin besar = semakin berpengaruh)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12.5),
    )
    return fig


def render_global_insight():
    """Panel Insight Global Model: Summary Plot + Bar Plot gabungan (Bagian 6.1 notebook)."""
    st.subheader("📊 Insight Global Model (SHAP)")
    st.caption(
        "Interpretasi ini menjelaskan pola umum yang dipelajari model Random Forest di seluruh "
        f"data (sampel acak {GLOBAL_SHAP_SAMPLE_SIZE} URL), bukan hanya satu URL tertentu — "
        "fitur apa yang paling berpengaruh secara umum, dan ke arah mana."
    )

    X_sample, shap_values_phishing = compute_global_shap(model, explainer)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**SHAP Summary Plot — Ringkasan Global**")
        st.caption(
            "Setiap titik satu sampel URL; posisi menunjukkan besar & arah kontribusi, warna "
            "menunjukkan nilai fitur asli (merah = tinggi, biru = rendah)."
        )
        render_global_summary_plot(X_sample, shap_values_phishing)
    with col2:
        st.markdown("**SHAP Bar Plot — Kepentingan Fitur (Gabungan)**")
        st.caption(
            "Rata-rata besaran (magnitude) kontribusi absolut tiap fitur terhadap prediksi, "
            "tanpa memandang arah pengaruhnya — semakin panjang batang, semakin berpengaruh."
        )
        st.plotly_chart(
            render_global_bar_plot(FEATURE_ORDER, shap_values_phishing),
            use_container_width=True,
        )


# ----------------------------------------------------------------------------
# Kalimat penjelasan XAI per fitur (kenapa fitur mendorong ke phishing/legit)
# ----------------------------------------------------------------------------
def build_reason_sentence(fname, value, shap_val):
    pushes_phishing = shap_val > 0
    v = int(value)

    templates = {
        "UrlLength": (
            f"URL cukup panjang ({v} karakter), pola yang sering dipakai untuk menyamarkan domain asli",
            f"panjang URL wajar ({v} karakter), sesuai pola tautan yang umum digunakan",
        ),
        "NumDots": (
            f"URL memiliki cukup banyak tanda titik ({v}), berpotensi menyembunyikan struktur domain sebenarnya",
            f"jumlah tanda titik pada URL wajar ({v})",
        ),
        "SubdomainLevel": (
            f"URL memiliki beberapa level subdomain ({v}), teknik umum untuk menyerupai domain resmi",
            f"struktur subdomain URL sederhana ({v} level)",
        ),
        "NumDash": (
            f"jumlah tanda hubung pada URL ini ({v}) sesuai pola yang pada dataset penelitian lebih sering ditemukan pada URL phishing",
            f"jumlah tanda hubung pada URL ini ({v}) sesuai pola yang pada dataset penelitian lebih sering ditemukan pada URL legitimate",
        ),
        "NumUnderscore": (
            f"jumlah underscore pada URL ini ({v}) sesuai pola yang pada dataset penelitian lebih sering ditemukan pada URL phishing",
            f"jumlah underscore pada URL ini ({v}) sesuai pola yang pada dataset penelitian lebih sering ditemukan pada URL legitimate",
        ),
        "NumPercent": (
            f"jumlah karakter encoding '%' pada URL ini ({v}) sesuai pola yang pada dataset penelitian lebih sering ditemukan pada URL phishing",
            f"jumlah karakter encoding '%' pada URL ini ({v}) sesuai pola yang pada dataset penelitian lebih sering ditemukan pada URL legitimate",
        ),
        "NumQueryComponents": (
            f"URL membawa cukup banyak parameter query ({v}), berpotensi menyisipkan data pengelabuan",
            f"jumlah parameter query pada URL wajar ({v})",
        ),
        "NumAmpersand": (
            f"URL memiliki beberapa simbol '&' ({v}) yang menandakan banyak parameter tersembunyi",
            f"jumlah simbol '&' pada URL wajar ({v})",
        ),
        "RandomString": (
            "nama domain terindikasi berupa string acak (mirip hasil generate otomatis / DGA)",
            "nama domain tidak menyerupai string acak",
        ),
        "NoHttps": (
            "URL tidak menggunakan protokol HTTPS yang aman",
            "URL sudah menggunakan protokol HTTPS",
        ),
        "DomainInPaths": (
            f"nama domain muncul kembali pada bagian path URL ({v}), pola yang teridentifikasi pada dataset penelitian",
            f"nama domain tidak diulang secara mencurigakan pada path URL ({v})",
        ),
        "DomainInSubdomains": (
            "nama domain resmi disisipkan sebagai subdomain (mis. 'paypal.secure-login.com'), pola khas phishing",
            "tidak ditemukan penyisipan nama domain pada subdomain",
        ),
        "HTTPSInHostname": (
            "kata 'https' disisipkan langsung di dalam hostname untuk membuat URL terlihat aman secara visual",
            "tidak ditemukan kata 'https' yang disisipkan mencurigakan pada hostname",
        ),
        "HostnameLength": (
            f"hostname URL cukup panjang ({v} karakter), pola umum saat domain palsu ditambahi kata-kata pengelabu",
            f"panjang hostname wajar ({v} karakter)",
        ),
    }
    high_text, low_text = templates.get(fname, (f"nilai fitur {fname} tinggi", f"nilai fitur {fname} wajar"))
    return high_text if pushes_phishing else low_text


# ----------------------------------------------------------------------------
# Fungsi inti: scan URL -> prediksi + SHAP
# ----------------------------------------------------------------------------
def scan_url(raw_url: str) -> dict:
    feats = extract_features_14(raw_url)
    X_row = pd.DataFrame([feats])[feature_cols]

    pred = int(model.predict(X_row)[0])
    proba_phishing = float(model.predict_proba(X_row)[0, 1])
    label = "Phishing" if pred == 1 else "Legitimate"

    sv = explainer.shap_values(X_row)
    sv_phishing = sv[1][0] if isinstance(sv, list) else sv[:, :, 1][0]
    base_value = get_base_value()

    contributions = []
    for fname, shap_val in zip(feature_cols, sv_phishing):
        meta = FEATURE_META[fname]
        contributions.append({
            "feature": fname,
            "no": meta["no"],
            "label": meta["label"],
            "desc": meta["desc"],
            "value": feats[fname],
            "shap": float(shap_val),
            "direction": "phishing" if shap_val > 0 else "legitimate",
            "reason": build_reason_sentence(fname, feats[fname], shap_val),
        })

    contributions_sorted = sorted(contributions, key=lambda c: abs(c["shap"]), reverse=True)
    top_reasons = [c for c in contributions_sorted if c["direction"] == label.lower()][:5]
    contributions_by_order = sorted(contributions, key=lambda c: FEATURE_ORDER.index(c["feature"]))

    return {
        "url": raw_url,
        "hostname": get_hostname(raw_url),
        "label": label,
        "proba_phishing": proba_phishing,
        "proba_legitimate": 1 - proba_phishing,
        "base_value": base_value,
        "features": contributions_by_order,
        "features_sorted": contributions_sorted,
        "top_reasons": top_reasons,
    }


# ----------------------------------------------------------------------------
# Komponen visual
# ----------------------------------------------------------------------------
def render_shap_chart(features_sorted):
    """Diverging bar chart kontribusi SHAP tiap fitur (merah=phishing, hijau=legitimate)."""
    top = list(reversed(features_sorted[:14]))
    labels = [f"{f['no']}. {f['label']}" for f in top]
    values = [f["shap"] for f in top]
    colors = ["#F4574B" if v > 0 else "#34D399" for v in values]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.3f}" for v in values],
            textposition="outside",
            hovertemplate="%{y}<br>Kontribusi SHAP: %{x:.4f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_width=1, line_color="#888")
    fig.update_layout(
        height=440,
        margin=dict(l=10, r=40, t=10, b=10),
        xaxis_title="← mendukung Legitimate     |     mendukung Phishing →",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12.5),
    )
    return fig


def render_result(result):
    is_phishing = result["label"] == "Phishing"

    st.divider()
    col_badge, col_url = st.columns([1, 3])
    with col_badge:
        if is_phishing:
            st.error("⚠️ **TERINDIKASI PHISHING**")
        else:
            st.success("✅ **TERINDIKASI LEGITIMATE**")
    with col_url:
        st.markdown(f"**URL:** `{result['url']}`")
        st.caption(f"hostname: {result['hostname']}")

    # --- probabilitas ---
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Probabilitas Phishing", f"{result['proba_phishing'] * 100:.1f}%")
        st.progress(result["proba_phishing"])
    with c2:
        st.metric("Probabilitas Legitimate", f"{result['proba_legitimate'] * 100:.1f}%")
        st.progress(result["proba_legitimate"])

    # --- alasan utama (XAI) ---
    st.subheader(
        "Kenapa URL ini terindikasi **Phishing**?" if is_phishing
        else "Kenapa URL ini terindikasi **Legitimate**?"
    )
    if result["top_reasons"]:
        for r in result["top_reasons"]:
            icon = "🔴" if r["direction"] == "phishing" else "🟢"
            st.markdown(f"{icon} **{r['label']}** — {r['reason']}")
    else:
        st.write("Tidak ada fitur dominan yang menonjol pada URL ini.")

    # --- grafik SHAP ---
    st.subheader("Kontribusi Setiap Fitur (SHAP)")
    st.caption(
        "Setiap batang menunjukkan seberapa besar & ke arah mana sebuah fitur mendorong prediksi model — "
        "merah mendorong ke **Phishing**, hijau mendorong ke **Legitimate**."
    )
    st.plotly_chart(render_shap_chart(result["features_sorted"]), use_container_width=True)

    # --- tabel detail 14 fitur ---
    with st.expander("Lihat tabel lengkap ke-14 fitur beserta nilai & kontribusinya"):
        df = pd.DataFrame(result["features"])[["no", "label", "desc", "value", "shap", "direction"]]
        df.columns = ["No", "Fitur", "Deskripsi", "Nilai", "Kontribusi SHAP", "Arah"]
        df["Arah"] = df["Arah"].map({"phishing": "→ Phishing", "legitimate": "→ Legitimate"})
        st.dataframe(df, use_container_width=True, hide_index=True)


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛡️ URL Sentinel")
    st.caption("Random Forest + Explainable AI (SHAP) — dijalankan sepenuhnya secara lokal.")
    st.markdown("---")
    st.markdown("**Tentang aplikasi**")
    st.write(
        "Aplikasi ini mengimplementasikan model Random Forest hasil penelitian "
        "*Pengembangan Model Random Forest Berbasis Explainable AI untuk Deteksi "
        "Ancaman Phishing pada URL* untuk memeriksa apakah sebuah URL tergolong "
        "**phishing** atau **legitimate**, lengkap dengan penjelasan XAI (SHAP) "
        "atas keputusan model."
    )
    st.markdown("---")
    st.markdown("**14 Fitur yang Dianalisis (Tabel 3)**")
    for f in FEATURE_ORDER:
        meta = FEATURE_META[f]
        st.caption(f"**{meta['no']}. {meta['label']}** — {meta['desc']}")
    st.markdown("---")
    st.caption("Tidak ada data URL yang dikirim ke server luar; seluruh proses berjalan di komputer ini.")
    st.markdown("---")
    with st.expander("⚠️ Keterbatasan model"):
        st.caption(
            "Model hanya menggunakan 14 fitur leksikal/struktural URL (Tabel 3) — "
            "tanpa reputasi domain, umur domain (WHOIS), atau analisis konten halaman. "
            "Untuk URL yang probabilitasnya mendekati 50%, hasil sebaiknya tidak "
            "dijadikan satu-satunya acuan keputusan. Lihat subbab 'Kesimpulan' pada "
            "notebook penelitian untuk saran pengembangan lanjutan."
        )


# ----------------------------------------------------------------------------
# Halaman utama
# ----------------------------------------------------------------------------
st.title("🛡️ URL Sentinel")
st.markdown(
    "##### Periksa sebuah URL — Random Forest memprediksi **Phishing** vs **Legitimate**, "
    "SHAP menjelaskan alasannya."
)

tab_scan, tab_global = st.tabs(["🔍 Periksa URL", "📊 Insight Global Model (SHAP)"])

with tab_scan:
    with st.form("scan_form"):
        url_input = st.text_input(
            "Masukkan URL yang ingin diperiksa",
            placeholder="mis. http://verify-account-update.tk/login?id=123",
        )
        submitted = st.form_submit_button("🔍 Jalankan Analisis", use_container_width=True)

    st.caption("Atau coba contoh:")
    ex1, ex2, ex3 = st.columns(3)
    example_clicked = None
    with ex1:
        if st.button("🟢 wikipedia.org", use_container_width=True):
            example_clicked = "https://www.wikipedia.org/wiki/Machine_learning"
    with ex2:
        if st.button("🔴 secure-login-update.tk", use_container_width=True):
            example_clicked = "http://secure-login-update.tk/verify/account?id=12345&token=x"
    with ex3:
        if st.button("🔴 https-paypal-secure...", use_container_width=True):
            example_clicked = "https://https-paypal-secure.verify-account123.com/login"

    target_url = None
    if submitted and url_input.strip():
        target_url = url_input.strip()
    elif example_clicked:
        target_url = example_clicked

    if target_url:
        with st.spinner("Mengekstrak fitur, menjalankan model, menghitung nilai SHAP ..."):
            try:
                result = scan_url(target_url)
                render_result(result)
            except Exception as e:
                st.error(f"Gagal memproses URL: {e}")
    elif submitted:
        st.warning("URL tidak boleh kosong.")

with tab_global:
    try:
        render_global_insight()
    except Exception as e:
        st.error(f"Gagal menghitung insight global: {e}")