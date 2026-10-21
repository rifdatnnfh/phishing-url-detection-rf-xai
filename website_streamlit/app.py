"""
URL Sentinel — Deteksi Phishing URL berbasis Random Forest + Explainable AI (SHAP)
Aplikasi web lokal menggunakan Streamlit.

Disinkronkan dengan notebook v5 (Seleksi Fitur Sebelum Ekstraksi):
32 fitur diekstraksi -> 19 fitur hasil permutation importance dipakai model.

Jalankan dengan:
    streamlit run app.py
"""
import re
from pathlib import Path
from urllib.parse import urlparse

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from features import FEATURE_META, extract_features, get_hostname

# Path dinamis berbasis lokasi file app.py menggunakan pathlib
RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "random_forest_phishing_model.joblib"
COLS_PATH = BASE_DIR / "model" / "feature_columns.joblib"

def _cari_dataset(nama):
    """Mencari dataset di folder aplikasi, lalu di ../notebook (lokasi dataset di repo).

    Memakai dataset_<nama>_1.csv bila ada, jika tidak dataset_<nama>.csv.
    """
    for folder in (BASE_DIR, BASE_DIR.parent / "notebook"):
        for fname in (f"dataset_{nama}_1.csv", f"dataset_{nama}.csv"):
            if (folder / fname).exists():
                return folder / fname
    return BASE_DIR / f"dataset_{nama}.csv"   # tidak ada: biarkan error FileNotFound yang jelas


DATASET_KAGGLE_PATH = _cari_dataset("kaggle")
DATASET_PHISHTANK_PATH = _cari_dataset("phishtank")

GLOBAL_SHAP_SAMPLE_SIZE = 200   # sama dengan jumlah sampel SHAP pada notebook

# ----------------------------------------------------------------------------
# Konfigurasi halaman
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="URL Sentinel — Deteksi Phishing",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Load model (di-cache agar tidak dimuat ulang setiap interaksi)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model Random Forest ...")
def load_model():
    model = joblib.load(MODEL_PATH)
    feature_cols = list(joblib.load(COLS_PATH))

    # Sumber kebenaran daftar fitur = feature_columns.joblib hasil notebook (19 fitur).
    unknown = [f for f in feature_cols if f not in FEATURE_META]
    if unknown:
        raise ValueError(
            f"feature_columns.joblib memuat fitur yang tidak dikenal features.py: {unknown}"
        )
    n_model = getattr(model, "n_features_in_", len(feature_cols))
    if n_model != len(feature_cols):
        raise ValueError(
            f"Model dilatih dengan {n_model} fitur, tetapi feature_columns.joblib berisi "
            f"{len(feature_cols)} fitur. Pastikan kedua file berasal dari notebook yang sama."
        )
    explainer = shap.TreeExplainer(model)
    return model, feature_cols, explainer


model, feature_cols, explainer = load_model()
N_FEATURES = len(feature_cols)
# Nomor urut tampilan mengikuti urutan fitur terpilih pada model
FEATURE_NO = {f: i + 1 for i, f in enumerate(feature_cols)}


def get_base_value():
    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        return float(ev[1]) if len(ev) > 1 else float(ev[0])
    return float(ev)


def feature_rows_to_frame(urls):
    """Ekstraksi fitur untuk banyak URL sekaligus -> DataFrame berkolom feature_cols."""
    return pd.DataFrame([extract_features(u) for u in urls])[feature_cols]


# ----------------------------------------------------------------------------
# Insight Global (SHAP) — mereproduksi Bagian 6.1 notebook penelitian:
# dataset dibersihkan dengan proses yang identik (Bagian 2 notebook), lalu diambil
# sampel untuk dihitung nilai SHAP-nya sehingga bisa ditampilkan Summary Plot (global)
# dan Bar Plot gabungan (rata-rata |SHAP value| per fitur) di aplikasi web.
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_clean_dataset():
    """Menggabungkan & membersihkan dataset Kaggle + PhishTank.

    Langkahnya identik dengan notebook penelitian (Bagian 2):
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
def compute_global_shap(_model, _explainer, feature_key, sample_size=GLOBAL_SHAP_SAMPLE_SIZE):
    """Mengambil sampel acak dari dataset, mengekstrak fitur, dan menghitung SHAP-nya.

    Parameter diawali underscore (_model, _explainer) agar Streamlit tidak
    mencoba melakukan hashing pada objek tersebut saat caching. `feature_key`
    (tuple nama fitur) ikut menjadi kunci cache supaya hasil lama otomatis tidak
    dipakai bila daftar fitur model berubah.
    """
    dataset = load_clean_dataset()
    n = min(sample_size, len(dataset))
    sample = dataset.sample(n=n, random_state=RANDOM_STATE).reset_index(drop=True)

    X_sample = feature_rows_to_frame(sample["url"])

    sv = _explainer.shap_values(X_sample)
    shap_values_phishing = sv[1] if isinstance(sv, list) else sv[:, :, 1]

    return X_sample, shap_values_phishing


def render_global_summary_plot(X_sample, shap_values_phishing):
    """SHAP Summary Plot (dot) ringkasan global kontribusi & arah tiap fitur."""
    plt.figure(figsize=(8, max(6, 0.32 * N_FEATURES)))
    shap.summary_plot(
        shap_values_phishing, X_sample, plot_type="dot",
        max_display=N_FEATURES, show=False, plot_size=None,
    )
    fig = plt.gcf()
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)


def render_global_bar_plot(shap_values_phishing):
    """Bar plot gabungan rata-rata SHAP value per fitur (kepentingan global fitur)."""
    mean_abs_shap = pd.Series(
        np.abs(shap_values_phishing).mean(axis=0), index=feature_cols
    ).sort_values(ascending=True)

    labels = [f"{FEATURE_NO[f]}. {FEATURE_META[f]['label']}" for f in mean_abs_shap.index]
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
        height=max(440, 28 * N_FEATURES),
        margin=dict(l=10, r=40, t=10, b=10),
        xaxis_title="Rata-rata |SHAP value| (semakin besar = semakin berpengaruh)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12.5),
    )
    return fig


def render_global_insight():
    """Panel Insight Global Model adalah Summary Plot + Bar Plot gabungan (Bagian 6.1 notebook)."""
    st.subheader("📊 Insight Global Model (SHAP)")
    st.caption(
        "Interpretasi ini menjelaskan pola umum yang dipelajari model Random Forest di seluruh "
        f"data (sampel acak {GLOBAL_SHAP_SAMPLE_SIZE} URL), bukan hanya satu URL tertentu — "
        "fitur apa yang paling berpengaruh secara umum, dan ke arah mana."
    )

    X_sample, shap_values_phishing = compute_global_shap(
        model, explainer, tuple(feature_cols)
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**SHAP Summary Plot - Ringkasan Global**")
        st.caption(
            "Setiap titik satu sampel URL, posisi menunjukkan besar & arah kontribusi, warna "
            "menunjukkan nilai fitur asli (merah = tinggi, biru = rendah)."
        )
        render_global_summary_plot(X_sample, shap_values_phishing)
    with col2:
        st.markdown("**SHAP Bar Plot - Kepentingan Fitur (Gabungan)**")
        st.caption(
            "Rata-rata besaran (magnitude) kontribusi absolut tiap fitur terhadap prediksi, "
            "tanpa memandang arah pengaruhnya — semakin panjang batang, semakin berpengaruh."
        )
        st.plotly_chart(
            render_global_bar_plot(shap_values_phishing),
            use_container_width=True,
        )


# ----------------------------------------------------------------------------
# Kalimat penjelasan XAI per fitur (kenapa fitur mendorong ke phishing/legit)
# Disamakan dengan build_reason_sentence pada notebook v5 (Bagian 7).
# ----------------------------------------------------------------------------
def build_reason_sentence(fname, value, shap_val):
    pushes_phishing = shap_val > 0
    v = int(value)
    vf = f"{float(value):.2f}"

    pola_phish = "sesuai pola yang pada dataset penelitian lebih sering ditemukan pada URL phishing"
    pola_legit = "sesuai pola yang pada dataset penelitian lebih sering ditemukan pada URL legitimate"

    templates = {
        "UrlLength": (
            f"URL cukup panjang ({v} karakter), pola yang sering dipakai untuk menyamarkan domain asli",
            f"panjang URL wajar ({v} karakter)",
        ),
        "NumDots": (
            f"URL memiliki cukup banyak tanda titik ({v})",
            f"jumlah tanda titik pada URL wajar ({v})",
        ),
        "SubdomainLevel": (
            f"URL memiliki beberapa level subdomain ({v}), teknik umum untuk menyerupai domain resmi",
            f"struktur subdomain URL sederhana ({v} level)",
        ),
        "NumDash": (
            f"jumlah tanda hubung pada URL ini ({v}) {pola_phish}",
            f"jumlah tanda hubung pada URL ini ({v}) {pola_legit}",
        ),
        "NumUnderscore": (
            f"jumlah underscore pada URL ini ({v}) {pola_phish}",
            f"jumlah underscore pada URL ini ({v}) {pola_legit}",
        ),
        "NumPercent": (
            f"jumlah karakter encoding '%' pada URL ini ({v}) {pola_phish}",
            f"jumlah karakter encoding '%' pada URL ini ({v}) {pola_legit}",
        ),
        "NumQueryComponents": (
            f"URL membawa cukup banyak parameter query ({v})",
            f"jumlah parameter query wajar ({v})",
        ),
        "NumAmpersand": (
            f"URL memiliki beberapa simbol '&' ({v})",
            f"jumlah simbol '&' wajar ({v})",
        ),
        "RandomString": (
            "nama domain terindikasi berupa string acak (mirip hasil generate otomatis)",
            "nama domain tidak menyerupai string acak",
        ),
        "NoHttps": (
            "URL tidak menggunakan protokol HTTPS yang aman",
            "URL sudah menggunakan protokol HTTPS",
        ),
        "DomainInPaths": (
            f"nama domain muncul kembali pada path URL ({v}), pola yang teridentifikasi pada dataset penelitian",
            f"nama domain tidak diulang secara mencurigakan pada path URL ({v})",
        ),
        "DomainInSubdomains": (
            "nama domain resmi disisipkan sebagai subdomain, pola khas phishing",
            "tidak ditemukan penyisipan nama domain pada subdomain",
        ),
        "HTTPSInHostname": (
            "kata 'https' disisipkan di dalam hostname agar URL terlihat aman secara visual",
            "tidak ditemukan kata 'https' yang disisipkan mencurigakan pada hostname",
        ),
        "HostnameLength": (
            f"hostname URL cukup panjang ({v} karakter)",
            f"panjang hostname wajar ({v} karakter)",
        ),
        "PathLength": (
            f"path URL cukup panjang ({v} karakter)",
            f"panjang path URL wajar ({v} karakter)",
        ),
        "QueryLength": (
            f"query string cukup panjang ({v} karakter)",
            f"panjang query string wajar ({v} karakter)",
        ),
        "NumSlash": (
            f"jumlah garis miring (/) pada URL ini ({v}) {pola_phish}",
            f"jumlah garis miring (/) pada URL ini ({v}) {pola_legit}",
        ),
        "NumDigits": (
            f"URL memuat cukup banyak angka ({v})",
            f"jumlah angka pada URL wajar ({v})",
        ),
        "DigitRatio": (
            f"proporsi angka pada URL cukup tinggi ({vf})",
            f"proporsi angka pada URL wajar ({vf})",
        ),
        "NumEquals": (
            f"URL memiliki beberapa simbol '=' ({v}) yang menandakan banyak parameter",
            f"jumlah simbol '=' pada URL wajar ({v})",
        ),
        "NumAt": (
            f"URL memuat simbol '@' ({v}), teknik yang dipakai untuk menyamarkan tujuan sebenarnya",
            f"tidak ada simbol '@' yang mencurigakan pada URL ({v})",
        ),
        "NumQuestion": (
            f"jumlah tanda tanya (?) pada URL ini ({v}) {pola_phish}",
            f"jumlah tanda tanya (?) pada URL ini ({v}) {pola_legit}",
        ),
        "HostDigits": (
            f"hostname memuat angka ({v}), pola yang sering muncul pada domain palsu",
            f"jumlah angka pada hostname wajar ({v})",
        ),
        "IsIP": (
            "hostname berupa alamat IP, bukan nama domain",
            "hostname berupa nama domain, bukan alamat IP",
        ),
        "NumKeywords": (
            f"URL memuat kata kunci sensitif seperti login/secure/verify ({v} kata)",
            f"tidak ada kata kunci sensitif yang menonjol pada URL ({v})",
        ),
        "EntropyUrl": (
            f"karakter URL terlihat cukup acak (entropi {vf})",
            f"pola karakter URL wajar (entropi {vf})",
        ),
        "EntropyHost": (
            f"karakter hostname terlihat cukup acak (entropi {vf})",
            f"pola karakter hostname wajar (entropi {vf})",
        ),
        "NumSpecial": (
            f"URL memuat cukup banyak karakter khusus ({v})",
            f"jumlah karakter khusus pada URL wajar ({v})",
        ),
        "HasPort": (
            "URL menuliskan nomor port secara eksplisit",
            "URL tidak menuliskan port yang tidak lazim",
        ),
        "NumUpper": (
            f"URL memuat huruf kapital ({v})",
            f"jumlah huruf kapital pada URL wajar ({v})",
        ),
        "HasWWW": (
            f"hostname tidak diawali 'www.', {pola_phish}" if v == 0
            else "hostname diawali 'www.' namun kombinasi fitur lain tetap mendorong ke arah phishing",
            f"hostname diawali 'www.', {pola_legit}" if v == 1
            else "hostname tidak diawali 'www.' namun kombinasi fitur lain tetap mendorong ke arah legitimate",
        ),
        "TLDLength": (
            f"panjang TLD ({v} karakter) {pola_phish}",
            f"panjang TLD ({v} karakter) {pola_legit}",
        ),
    }
    high_text, low_text = templates.get(fname, (f"nilai fitur {fname} tinggi", f"nilai fitur {fname} wajar"))
    return high_text if pushes_phishing else low_text


# ----------------------------------------------------------------------------
# Normalisasi URL sebelum ekstraksi fitur
# ----------------------------------------------------------------------------
def normalize_url(raw_url: str) -> str:
    """Menyeragamkan penulisan URL sebelum fitur diekstrak.

    Garis miring (/) di akhir path tidak mengubah tujuan URL
    (https://www.youtube.com/ dan https://www.youtube.com adalah situs yang sama),
    tetapi menambah 1 karakter pada fitur UrlLength sehingga model Random Forest
    bisa memberi hasil berbeda untuk dua URL yang sama. Karena itu, garis miring
    penutup dibuang terlebih dahulu. Skema (http/https) dan bagian query/fragment
    tidak diubah, jadi fitur NoHttps dan fitur lainnya tetap dihitung apa adanya.
    """
    url = str(raw_url).strip()
    m = re.match(r"^([^?#]*)(.*)$", url, flags=re.S)
    base, tail = m.group(1), m.group(2)
    if "://" in base:
        scheme, rest = base.split("://", 1)
        base = scheme + "://" + rest.rstrip("/")
    else:
        base = base.rstrip("/")
    return base + tail


# ----------------------------------------------------------------------------
# Validasi input URL
# ----------------------------------------------------------------------------
class InvalidURLError(ValueError):
    """Dilempar saat input bukan URL yang valid (ditolak sebelum masuk model)."""


MAX_URL_LENGTH = 2048
_IPV4_RE = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")


def validate_url(raw_url):
    """Validasi format URL SEBELUM diproses model.

    Mengembalikan tuple (valid, alasan). Input dianggap tidak valid jika:
    kosong, terlalu panjang, mengandung spasi, skema bukan http/https,
    tidak punya hostname, atau hostname tidak berbentuk domain/IPv4 yang wajar.
    URL tanpa skema (mis. 'www.bbc.com/news') tetap diterima; pada ekstraksi fitur,
    URL tanpa skema dianggap HTTPS (fitur NoHttps = 0), sesuai notebook.
    """
    if raw_url is None or not str(raw_url).strip():
        return False, "input kosong"
    url = str(raw_url).strip()

    if len(url) > MAX_URL_LENGTH:
        return False, f"URL terlalu panjang (> {MAX_URL_LENGTH} karakter)"
    if re.search(r"\s", url):
        return False, "URL mengandung spasi"

    if "://" in url:
        scheme = url.split("://", 1)[0].lower()
        if scheme not in ("http", "https"):
            return False, f"skema '{scheme}://' tidak didukung (hanya http/https)"
        candidate = url
    else:
        candidate = "http://" + url

    try:
        host = urlparse(candidate).hostname
    except ValueError:
        return False, "format URL tidak dapat dibaca"
    if not host:
        return False, "hostname/domain tidak ditemukan"
    host = host.rstrip(".")
    if len(host) > 253:
        return False, "hostname terlalu panjang"

    if _IPV4_RE.match(host):
        if all(0 <= int(p) <= 255 for p in host.split(".")):
            return True, "ok"
        return False, "alamat IP tidak valid"

    labels = host.split(".")
    if len(labels) < 2:
        return False, "hostname tidak berbentuk domain (tidak ada titik/TLD)"
    if not all(re.fullmatch(r"[\w-]{1,63}", lb) for lb in labels):
        return False, "hostname mengandung karakter atau label yang tidak valid"
    if not re.fullmatch(r"xn--[\w-]+|[^\W\d_]{2,}", labels[-1]):
        return False, "TLD tidak valid"
    return True, "ok"


# ----------------------------------------------------------------------------
# Fungsi inti: scan URL -> prediksi + SHAP
# ----------------------------------------------------------------------------
def scan_url(raw_url: str) -> dict:
    # tolak input tidak valid sebelum masuk model
    valid, alasan = validate_url(raw_url)
    if not valid:
        raise InvalidURLError(alasan)

    url_normalized = normalize_url(raw_url)
    feats = extract_features(url_normalized)          # 32 fitur
    X_row = pd.DataFrame([feats])[feature_cols]       # dipilih 19 fitur model

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
            "no": FEATURE_NO[fname],
            "label": meta["label"],
            "desc": meta["desc"],
            "value": feats[fname],
            "shap": float(shap_val),
            "direction": "phishing" if shap_val > 0 else "legitimate",
            "reason": build_reason_sentence(fname, feats[fname], shap_val),
        })

    contributions_sorted = sorted(contributions, key=lambda c: abs(c["shap"]), reverse=True)
    top_reasons = [c for c in contributions_sorted if c["direction"] == label.lower()][:5]
    contributions_by_order = sorted(contributions, key=lambda c: c["no"])

    return {
        "url": raw_url,
        "url_normalized": url_normalized,
        "hostname": get_hostname(url_normalized),
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
    top = list(reversed(features_sorted))
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
        height=max(440, 28 * len(top)),
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
        if result["url_normalized"] != result["url"].strip():
            st.caption(
                f"URL dinormalisasi sebelum dianalisis (garis miring di akhir dibuang): "
                f"`{result['url_normalized']}`"
            )

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

    # --- tabel detail seluruh fitur ---
    with st.expander(f"Lihat tabel lengkap ke-{N_FEATURES} fitur beserta nilai & kontribusinya"):
        df = pd.DataFrame(result["features"])[["no", "label", "desc", "value", "shap", "direction"]]
        df.columns = ["No", "Fitur", "Deskripsi", "Nilai", "Kontribusi SHAP", "Arah"]
        df["Arah"] = df["Arah"].map({"phishing": "→ Phishing", "legitimate": "→ Legitimate"})
        st.dataframe(df, use_container_width=True, hide_index=True)


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛡️ URL Sentinel")
    st.caption("Random Forest + Explainable AI (SHAP)")
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
    st.markdown(f"**{N_FEATURES} Fitur yang Dianalisis**")
    st.caption(
        "Hasil seleksi fitur (permutation importance) dari 32 kandidat fitur leksikal URL."
    )
    for f in feature_cols:
        meta = FEATURE_META[f]
        st.caption(f"**{FEATURE_NO[f]}. {meta['label']}** — {meta['desc']}")
    st.markdown("---")
    st.caption("Tidak ada data URL yang dikirim ke server luar.")
    st.markdown("---")
    with st.expander("⚠️ Keterbatasan model"):
        st.caption(
            f"Model hanya menggunakan {N_FEATURES} fitur leksikal atau struktural URL "
            "tanpa reputasi domain, umur domain (WHOIS), atau analisis konten halaman."
        )


# ----------------------------------------------------------------------------
# Halaman utama
# ----------------------------------------------------------------------------
st.title("🛡️ URL Sentinel")
st.markdown(
    "##### Periksa sebuah URL\n\n"
    "Random Forest memprediksi **Phishing** vs **Legitimate**, "
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
            except InvalidURLError as e:
                st.warning(
                    f"⚠️ Input tidak valid dan tidak diproses oleh model: {e}. "
                    "Masukkan URL lengkap, mis. https://contoh.com/halaman"
                )
            except Exception as e:
                st.error(f"Gagal memproses URL: {e}")
    elif submitted:
        st.warning("URL tidak boleh kosong.")

with tab_global:
    try:
        render_global_insight()
    except Exception as e:
        st.error(f"Gagal menghitung insight global: {e}")