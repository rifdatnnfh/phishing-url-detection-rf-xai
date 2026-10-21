"""
Ekstraksi fitur URL untuk URL Sentinel.

Isi file ini disamakan dengan notebook `Deteksi_Phishing_RF_XAI.ipynb`:
32 fitur leksikal diekstraksi (14 fitur dasar + 18 fitur tambahan). Model akhir
hanya memakai sebagian fitur hasil seleksi (permutation importance); daftar fitur
yang dipakai model dibaca dari model/feature_columns.joblib oleh app.py.
"""
import math
import re
from collections import Counter
from urllib.parse import urlparse

BASE_FEATURES = [
    "UrlLength", "NumDots", "SubdomainLevel", "NumDash", "NumUnderscore",
    "NumPercent", "NumQueryComponents", "NumAmpersand", "RandomString",
    "NoHttps", "DomainInPaths", "DomainInSubdomains", "HTTPSInHostname",
    "HostnameLength",
]
EXTRA_FEATURES = [
    "PathLength", "QueryLength", "NumSlash", "NumDigits", "DigitRatio",
    "NumEquals", "NumAt", "NumQuestion", "HostDigits", "IsIP", "NumKeywords",
    "EntropyUrl", "EntropyHost", "NumSpecial", "HasPort", "NumUpper", "HasWWW",
    "TLDLength",
]
FEATURE_ORDER = BASE_FEATURES + EXTRA_FEATURES

SENSITIVE_KEYWORDS = ["login", "secure", "verify", "account", "update", "bank",
                      "signin", "confirm", "paypal", "password", "webscr", "ebay"]

# Label & deskripsi setiap fitur (dipakai untuk tampilan dan narasi penjelasan XAI)
_FEATURE_TEXT = {
    "UrlLength":          ("Panjang URL", "Panjang keseluruhan string URL"),
    "NumDots":            ("Jumlah Titik (.)", "Jumlah tanda titik dalam URL"),
    "SubdomainLevel":     ("Tingkat Subdomain", "Jumlah level subdomain pada hostname"),
    "NumDash":            ("Jumlah Tanda Hubung (-)", "Jumlah karakter '-' dalam URL"),
    "NumUnderscore":      ("Jumlah Underscore (_)", "Jumlah karakter '_' dalam URL"),
    "NumPercent":         ("Jumlah Persen (%)", "Jumlah karakter '%' (encoding) dalam URL"),
    "NumQueryComponents": ("Jumlah Parameter Query", "Jumlah komponen parameter pada query string"),
    "NumAmpersand":       ("Jumlah Ampersand (&)", "Jumlah karakter '&' dalam URL"),
    "RandomString":       ("Indikasi String Acak",
                           "Deteksi pola nama domain yang menyerupai string acak (DGA-like)"),
    "NoHttps":            ("Tanpa HTTPS", "URL tidak menggunakan protokol HTTPS"),
    "DomainInPaths":      ("Domain Muncul di Path", "Nama domain muncul kembali pada bagian path URL"),
    "DomainInSubdomains": ("Domain Muncul di Subdomain",
                           "Nama domain muncul di bagian subdomain (trik penyamaran)"),
    "HTTPSInHostname":    ("Kata 'https' di Hostname",
                           "Kata 'https' disisipkan di dalam hostname untuk terlihat aman"),
    "HostnameLength":     ("Panjang Hostname", "Panjang nama host (domain + subdomain)"),
    # --- fitur tambahan ---
    "PathLength":         ("Panjang Path", "Panjang bagian path URL"),
    "QueryLength":        ("Panjang Query", "Panjang bagian query string URL"),
    "NumSlash":           ("Jumlah Garis Miring (/)", "Jumlah karakter '/' pada URL (tanpa skema)"),
    "NumDigits":          ("Jumlah Digit", "Jumlah karakter angka dalam URL"),
    "DigitRatio":         ("Rasio Digit", "Proporsi karakter angka terhadap panjang URL"),
    "NumEquals":          ("Jumlah Tanda Sama Dengan (=)", "Jumlah karakter '=' dalam URL"),
    "NumAt":              ("Jumlah Simbol @", "Jumlah karakter '@' dalam URL"),
    "NumQuestion":        ("Jumlah Tanda Tanya (?)", "Jumlah karakter '?' dalam URL"),
    "HostDigits":         ("Jumlah Digit di Hostname", "Jumlah karakter angka pada hostname"),
    "IsIP":               ("Hostname Berupa IP", "Hostname berupa alamat IPv4, bukan nama domain"),
    "NumKeywords":        ("Jumlah Kata Kunci Sensitif",
                           "Jumlah kata kunci sensitif (login, secure, verify, dll.) dalam URL"),
    "EntropyUrl":         ("Entropi URL", "Entropi Shannon karakter URL (tingkat keacakan)"),
    "EntropyHost":        ("Entropi Hostname", "Entropi Shannon karakter hostname (tingkat keacakan)"),
    "NumSpecial":         ("Jumlah Karakter Khusus", "Jumlah karakter non-alfanumerik dalam URL"),
    "HasPort":            ("Port Eksplisit", "URL menuliskan nomor port secara eksplisit"),
    "NumUpper":           ("Jumlah Huruf Kapital", "Jumlah huruf kapital dalam URL"),
    "HasWWW":             ("Hostname Diawali www", "Hostname diawali 'www.'"),
    "TLDLength":          ("Panjang TLD", "Panjang ekstensi domain tingkat atas (TLD)"),
}

FEATURE_META = {
    f: {"no": i + 1, "label": _FEATURE_TEXT[f][0], "desc": _FEATURE_TEXT[f][1]}
    for i, f in enumerate(FEATURE_ORDER)
}


def get_hostname(url: str) -> str:
    """Mengambil hostname (tanpa userinfo & port) dari sebuah URL."""
    url = str(url)
    parsed = urlparse(url if "://" in url else "http://" + url)
    hostname = parsed.netloc if parsed.netloc else url.split("/")[0]
    hostname = hostname.split("@")[-1]   # buang userinfo (user:pass@) jika ada
    hostname = hostname.split(":")[0]    # buang port jika ada
    return hostname.lower()


def get_domain_name(hostname: str) -> str:
    """Mengambil nama domain level-2 (tanpa subdomain & TLD)."""
    parts = hostname.split(".")
    if len(parts) >= 2:
        return parts[-2]
    return hostname


def is_random_string(s: str) -> int:
    """Heuristik RandomString: rasio vokal rendah / rangkaian konsonan panjang."""
    s = re.sub(r"[^a-zA-Z]", "", s)
    if len(s) < 4:
        return 0
    vowels = sum(1 for c in s.lower() if c in "aeiou")
    ratio = vowels / len(s)
    consonant_runs = re.findall(r"[^aeiou\d_]+", s.lower())
    max_run = max((len(r) for r in consonant_runs), default=0)
    return 1 if (ratio < 0.2 or max_run >= 5) else 0


def infer_no_https(url: str) -> int:
    """NoHttps: 1 = http://, 0 = https://. Skema tidak ditulis -> diasumsikan HTTPS."""
    u = url.lower()
    if u.startswith("https://"):
        return 0
    if u.startswith("http://"):
        return 1
    return 0


def shannon_entropy(s: str) -> float:
    """Entropi Shannon (bit per karakter) dari sebuah string."""
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _has_port(parsed) -> int:
    try:
        return 1 if parsed.port is not None else 0
    except ValueError:
        return 0


def extract_features(url: str) -> dict:
    """Mengekstrak seluruh (32) fitur pada FEATURE_ORDER dari sebuah URL."""
    url = str(url).strip()
    parsed = urlparse(url if "://" in url else "http://" + url)
    hostname = get_hostname(url)
    path = parsed.path
    query = parsed.query
    domain_name = get_domain_name(hostname)

    host_parts = hostname.split(".")
    subdomain_part = ".".join(host_parts[:-2]) if len(host_parts) > 2 else ""

    # URL tanpa skema: dipakai agar fitur tambahan tidak bergantung pada ada/tidaknya "http(s)://"
    no_scheme = url.split("://", 1)[1] if "://" in url else url
    low = url.lower()

    feats = {}
    # --- 14 fitur dasar ---
    feats["UrlLength"] = len(url)
    feats["NumDots"] = url.count(".")
    feats["SubdomainLevel"] = max(hostname.count(".") - 1, 0)
    feats["NumDash"] = url.count("-")
    feats["NumUnderscore"] = url.count("_")
    feats["NumPercent"] = url.count("%")
    feats["NumQueryComponents"] = len([q for q in query.split("&") if q != ""]) if query else 0
    feats["NumAmpersand"] = url.count("&")
    feats["RandomString"] = is_random_string(domain_name)
    feats["NoHttps"] = infer_no_https(url)
    feats["DomainInPaths"] = 1 if domain_name and domain_name in path.lower() else 0
    feats["DomainInSubdomains"] = 1 if domain_name and domain_name in subdomain_part.lower() else 0
    feats["HTTPSInHostname"] = 1 if "https" in hostname else 0
    feats["HostnameLength"] = len(hostname)
    # --- fitur tambahan ---
    feats["PathLength"] = len(path)
    feats["QueryLength"] = len(query)
    feats["NumSlash"] = no_scheme.count("/")
    feats["NumDigits"] = sum(ch.isdigit() for ch in url)
    feats["DigitRatio"] = sum(ch.isdigit() for ch in no_scheme) / max(len(no_scheme), 1)
    feats["NumEquals"] = url.count("=")
    feats["NumAt"] = url.count("@")
    feats["NumQuestion"] = url.count("?")
    feats["HostDigits"] = sum(ch.isdigit() for ch in hostname)
    feats["IsIP"] = 1 if re.fullmatch(r"(\d{1,3}\.){3}\d{1,3}", hostname) else 0
    feats["NumKeywords"] = sum(k in low for k in SENSITIVE_KEYWORDS)
    feats["EntropyUrl"] = shannon_entropy(no_scheme)
    feats["EntropyHost"] = shannon_entropy(hostname)
    feats["NumSpecial"] = sum(not ch.isalnum() for ch in no_scheme)
    feats["HasPort"] = _has_port(parsed)
    feats["NumUpper"] = sum(ch.isupper() for ch in no_scheme)
    feats["HasWWW"] = 1 if hostname.startswith("www.") else 0
    feats["TLDLength"] = len(host_parts[-1]) if hostname else 0

    return {f: feats[f] for f in FEATURE_ORDER}