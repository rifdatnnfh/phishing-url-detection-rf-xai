import re
from urllib.parse import urlparse

FEATURE_ORDER = [
    "UrlLength", "NumDots", "SubdomainLevel", "NumDash", "NumUnderscore",
    "NumPercent", "NumQueryComponents", "NumAmpersand", "RandomString",
    "NoHttps", "DomainInPaths", "DomainInSubdomains", "HTTPSInHostname",
    "HostnameLength",
]

# Deskripsi & metadata setiap fitur (dipakai untuk narasi penjelasan XAI)
FEATURE_META = {
    "UrlLength":          {"no": 1,  "label": "Panjang URL",
                            "desc": "Panjang keseluruhan string URL"},
    "NumDots":             {"no": 2,  "label": "Jumlah Titik (.)",
                            "desc": "Jumlah tanda titik dalam URL"},
    "SubdomainLevel":      {"no": 3,  "label": "Tingkat Subdomain",
                            "desc": "Jumlah level subdomain pada hostname"},
    "NumDash":              {"no": 4,  "label": "Jumlah Tanda Hubung (-)",
                            "desc": "Jumlah karakter '-' dalam URL"},
    "NumUnderscore":        {"no": 5,  "label": "Jumlah Underscore (_)",
                            "desc": "Jumlah karakter '_' dalam URL"},
    "NumPercent":           {"no": 6,  "label": "Jumlah Persen (%)",
                            "desc": "Jumlah karakter '%' (encoding) dalam URL"},
    "NumQueryComponents":   {"no": 7,  "label": "Jumlah Parameter Query",
                            "desc": "Jumlah komponen parameter pada query string"},
    "NumAmpersand":         {"no": 8,  "label": "Jumlah Ampersand (&)",
                            "desc": "Jumlah karakter '&' dalam URL"},
    "RandomString":         {"no": 9,  "label": "Indikasi String Acak",
                            "desc": "Deteksi pola nama domain yang menyerupai string acak (DGA-like)"},
    "NoHttps":              {"no": 10, "label": "Tanpa HTTPS",
                            "desc": "URL tidak menggunakan protokol HTTPS"},
    "DomainInPaths":        {"no": 11, "label": "Domain Muncul di Path",
                            "desc": "Nama domain muncul kembali pada bagian path URL"},
    "DomainInSubdomains":   {"no": 12, "label": "Domain Muncul di Subdomain",
                            "desc": "Nama domain muncul di bagian subdomain (trik penyamaran)"},
    "HTTPSInHostname":      {"no": 13, "label": "Kata 'https' di Hostname",
                            "desc": "Kata 'https' disisipkan di dalam hostname untuk terlihat aman"},
    "HostnameLength":       {"no": 14, "label": "Panjang Hostname",
                            "desc": "Panjang nama host (domain + subdomain)"},
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
    u = url.lower()
    if u.startswith("https://"):
        return 0
    if u.startswith("http://"):
        return 1
    return 0  # skema tidak dituliskan -> asumsikan HTTPS (standar web modern)


def extract_features_14(url: str) -> dict:
    """Mengekstrak 14 fitur URL sebagaimana didefinisikan pada Tabel 6."""
    url = str(url).strip()
    parsed = urlparse(url if "://" in url else "http://" + url)
    hostname = get_hostname(url)
    path = parsed.path
    query = parsed.query
    domain_name = get_domain_name(hostname)

    host_parts = hostname.split(".")
    subdomain_part = ".".join(host_parts[:-2]) if len(host_parts) > 2 else ""

    feats = {}
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
    return feats
