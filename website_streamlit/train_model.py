"""
Melatih ulang model Random Forest untuk deteksi phishing URL. Skrip ini
mereproduksi pipeline pada notebook `Deteksi_Phishing_RF_XAI.ipynb`:

    1. gabung & bersihkan dataset (Kaggle + PhishTank)
    2. seleksi fitur (permutation importance + binary search) dari 32 fitur
    3. ekstraksi fitur terpilih, split 70/15/15 (stratified)
    4. augmentasi data latih (halaman utama legitimate + invarian www)
    5. GridSearchCV pada subsampel, latih ulang pada seluruh data latih
    6. evaluasi pada data uji, simpan model ke folder model/

Jalankan sekali di awal (atau saat dataset berubah):
    python train_model.py
"""
import time
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
from features import FEATURE_ORDER, extract_features, get_hostname
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, train_test_split

RANDOM_STATE = 42
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"


def _dataset_path(nama):
    """Cari dataset di folder ini, lalu di ../notebook. Utamakan versi _1 bila ada."""
    for folder in (BASE_DIR, BASE_DIR.parent / "notebook"):
        for fname in (f"dataset_{nama}_1.csv", f"dataset_{nama}.csv"):
            if (folder / fname).exists():
                return folder / fname
    return BASE_DIR / f"dataset_{nama}.csv"


# --- Parameter seleksi fitur (sama dengan notebook) ---
PI_THRESHOLD = 0.001
MAX_DROP_F1 = 0.005
MIN_METRIK = 0.90
N_SEL = 150000
N_PI = 30000
N_TREES_SEL = 100

# --- Parameter augmentasi & tuning (sama dengan notebook) ---
PHISHING_WEIGHT_MULT = 1.3
N_TUNE = 250000
PARAM_GRID = {"n_estimators": [100, 200], "max_depth": [10, 20]}


def muat_dataset():
    kaggle = pd.read_csv(_dataset_path("kaggle"))
    phishtank = pd.read_csv(_dataset_path("phishtank"))
    kaggle["source"] = "Kaggle"
    phishtank["source"] = "PhishTank"

    dataset = pd.concat([kaggle, phishtank], ignore_index=True)
    dataset = dataset.drop_duplicates(subset="url")
    dataset["label"] = dataset["type"].map({"legitimate": 0, "phishing": 1})
    dataset = dataset.dropna(subset=["label"]).reset_index(drop=True)
    dataset["label"] = dataset["label"].astype(int)
    print("Ukuran dataset:", dataset.shape)
    return dataset


def seleksi_fitur(dataset):
    """Permutation importance + binary search (Bagian seleksi fitur notebook)."""
    full_features = list(FEATURE_ORDER)

    idx_train_utama, _ = train_test_split(
        dataset.index, test_size=0.3, random_state=RANDOM_STATE, stratify=dataset["label"]
    )
    sel = dataset.loc[idx_train_utama]
    if len(sel) > N_SEL:
        sel = sel.sample(N_SEL, random_state=RANDOM_STATE)

    F = pd.DataFrame([extract_features(u) for u in sel["url"]], index=sel.index)[full_features]
    y = sel["label"]
    Xtr, Xva, ytr, yva = train_test_split(
        F, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    def latih_sementara(cols):
        m = RandomForestClassifier(
            n_estimators=N_TREES_SEL, max_depth=20, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        )
        m.fit(Xtr[cols], ytr)
        return m

    def metrik_val(m, cols):
        p = m.predict_proba(Xva[cols])[:, 1]
        pr = (p >= 0.5).astype(int)
        return {
            "Accuracy": accuracy_score(yva, pr), "Precision": precision_score(yva, pr),
            "Recall": recall_score(yva, pr), "F1": f1_score(yva, pr),
            "ROC-AUC": roc_auc_score(yva, p),
        }

    model_sel = latih_sementara(full_features)
    metrik_base = metrik_val(model_sel, full_features)
    print("Metrik validasi-seleksi (32 fitur):", {k: round(v, 4) for k, v in metrik_base.items()})

    if len(Xva) > N_PI:
        X_pi, _, y_pi, _ = train_test_split(
            Xva, yva, train_size=N_PI, stratify=yva, random_state=RANDOM_STATE
        )
    else:
        X_pi, y_pi = Xva, yva

    pi = permutation_importance(
        model_sel, X_pi, y_pi, scoring={"f1": "f1", "roc_auc": "roc_auc"},
        n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1,
    )
    pi_df = pd.DataFrame({
        "F1_turun_mean": pi["f1"].importances_mean,
        "AUC_turun_mean": pi["roc_auc"].importances_mean,
    }, index=full_features).sort_values("F1_turun_mean", ascending=False)

    kandidat = pi_df[(pi_df["F1_turun_mean"] < PI_THRESHOLD) & (pi_df["AUC_turun_mean"] < PI_THRESHOLD)]
    kandidat = kandidat.sort_values(["F1_turun_mean", "AUC_turun_mean"]).index.tolist()
    print(f"Kandidat fitur tidak berpengaruh ({len(kandidat)}):", kandidat)

    def lolos(m, cols):
        mt = metrik_val(m, cols)
        ok = (metrik_base["F1"] - mt["F1"] <= MAX_DROP_F1) and all(
            v >= MIN_METRIK or metrik_base[k] < MIN_METRIK for k, v in mt.items()
        )
        return ok

    lo, hi = 0, len(kandidat)
    terbaik_cols = full_features
    while lo < hi:
        mid = (lo + hi + 1) // 2
        cols = [f for f in full_features if f not in kandidat[:mid]]
        if lolos(latih_sementara(cols), cols):
            lo, terbaik_cols = mid, cols
        else:
            hi = mid - 1

    feature_selected = list(terbaik_cols)
    print(f"Fitur terpilih ({len(feature_selected)}):", feature_selected)
    return feature_selected


def buat_augmentasi(data_final, X_train, y_train, feature_cols):
    """Augmentasi data latih: halaman utama legitimate + invarian www."""
    train_urls = data_final.loc[X_train.index, "url"].reset_index(drop=True)
    train_labels = y_train.reset_index(drop=True)
    train_www = pd.Series([1 if get_hostname(u).startswith("www.") else 0 for u in train_urls])

    netloc = np.array([urlparse(u).netloc.lower() for u in train_urls])
    scheme = np.array([u.split("://")[0] if "://" in u else "http" for u in train_urls])
    is_legit = (train_labels.values == 0)

    # (1) Halaman utama dari host legitimate pada data latih
    first_scheme = {h: s for h, s in zip(netloc[is_legit], scheme[is_legit])}
    hosts_legit_train = np.array(sorted(set(netloc[is_legit])))
    rng_aug = np.random.RandomState(0)
    sel_hosts = rng_aug.choice(
        hosts_legit_train, size=min(60000, len(hosts_legit_train)), replace=False
    )
    aug_home_urls = []
    for h in sel_hosts:
        s = first_scheme[h]
        aug_home_urls += [f"{s}://{h}", f"{s}://{h}/"]

    # (2) Invarian www
    tr_df = pd.DataFrame({"url": train_urls, "label": train_labels, "www": train_www})
    legit_www = tr_df[(tr_df.label == 0) & (tr_df.www == 1)]
    legit_www = legit_www.sample(min(50000, len(legit_www)), random_state=1)["url"]
    legit_www_off = legit_www.str.replace("://www.", "://", n=1, regex=False)
    phish_nowww = tr_df[(tr_df.label == 1) & (tr_df.www == 0)]
    phish_nowww = phish_nowww.sample(min(50000, len(phish_nowww)), random_state=1)["url"]
    phish_www_on = phish_nowww.apply(lambda u: u.split("://", 1)[0] + "://www." + u.split("://", 1)[1])

    aug_urls = list(aug_home_urls) + list(legit_www_off) + list(phish_www_on)
    aug_labels = np.array([0] * (len(aug_home_urls) + len(legit_www_off)) + [1] * len(phish_www_on))
    aug_X = pd.DataFrame([extract_features(u) for u in aug_urls])[feature_cols]

    X_fit = pd.concat([X_train, aug_X], ignore_index=True)
    y_fit = pd.concat([y_train.reset_index(drop=True), pd.Series(aug_labels)], ignore_index=True)

    n_fit = len(y_fit)
    n1 = int(y_fit.sum())
    n0 = n_fit - n1
    class_weight = {0: n_fit / (2 * n0), 1: PHISHING_WEIGHT_MULT * n_fit / (2 * n1)}
    print("Data latih asli:", len(X_train), "| setelah augmentasi:", n_fit)
    print("Bobot kelas:", {k: round(v, 3) for k, v in class_weight.items()})
    return X_fit, y_fit, class_weight


def main():
    print("Memuat & menggabungkan dataset ...")
    dataset = muat_dataset()

    print("\nSeleksi fitur (permutation importance) ...")
    feature_cols = seleksi_fitur(dataset)

    print("\nMengekstrak fitur terpilih untuk seluruh dataset ...")
    feature_df = pd.DataFrame(
        [extract_features(u) for u in dataset["url"]], index=dataset.index
    )[feature_cols]
    data_final = pd.concat([dataset[["url", "label"]], feature_df], axis=1)
    data_final = data_final.dropna().reset_index(drop=True) if data_final.isnull().any().any() else data_final

    X = data_final[feature_cols]
    y = data_final["label"]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
    )
    print("Data latih:", X_train.shape, "| Validasi:", X_val.shape, "| Uji:", X_test.shape)

    print("\nAugmentasi data latih ...")
    X_fit, y_fit, class_weight = buat_augmentasi(data_final, X_train, y_train, feature_cols)
    X_fit32 = X_fit.astype(np.float32)
    y_fit8 = y_fit.astype(np.int8)

    if len(X_fit32) > N_TUNE:
        X_tune, _, y_tune, _ = train_test_split(
            X_fit32, y_fit8, train_size=N_TUNE, stratify=y_fit8, random_state=RANDOM_STATE
        )
    else:
        X_tune, y_tune = X_fit32, y_fit8

    print("\nMenjalankan Grid Search CV ...")
    t0 = time.time()
    grid = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE, class_weight=class_weight, n_jobs=-1),
        PARAM_GRID, cv=5, scoring="f1", n_jobs=1, verbose=2,
    )
    grid.fit(X_tune, y_tune)
    print(f"Selesai dalam {time.time() - t0:.1f} detik | parameter terbaik: {grid.best_params_}")

    model = RandomForestClassifier(
        **grid.best_params_, random_state=RANDOM_STATE, class_weight=class_weight, n_jobs=-1
    )
    model.fit(X_fit32, y_fit8)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print("\n=== Evaluasi pada data uji ===")
    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score  : {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC   : {roc_auc_score(y_test, y_proba):.4f}")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "random_forest_phishing_model.joblib")
    joblib.dump(feature_cols, MODEL_DIR / "feature_columns.joblib")
    print("\nModel tersimpan di", MODEL_DIR / "random_forest_phishing_model.joblib")


if __name__ == "__main__":
    main()