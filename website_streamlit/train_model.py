"""
Melatih ulang model Random Forest untuk deteksi phishing URL, mereproduksi
pipeline pada notebook `Deteksi_Phishing_RF_XAI_Bab3.ipynb` (Bab III).

Jalankan sekali di awal (atau saat dataset berubah):
    python train_model.py
"""
import time
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

from features import FEATURE_ORDER, extract_features_14

RANDOM_STATE = 42


def main():
    print("Memuat & menggabungkan dataset ...")
    kaggle = pd.read_csv("dataset_kaggle.csv")
    phishtank = pd.read_csv("dataset_phishtank.csv")

    kaggle["source"] = "Kaggle"
    phishtank["source"] = "PhishTank"

    dataset = pd.concat([kaggle, phishtank], ignore_index=True)
    dataset = dataset.drop_duplicates(subset="url")
    dataset["label"] = dataset["type"].map({"legitimate": 0, "phishing": 1})
    dataset = dataset.dropna(subset=["label"]).reset_index(drop=True)
    dataset["label"] = dataset["label"].astype(int)
    print("Ukuran dataset:", dataset.shape)

    print("Mengekstrak 14 fitur URL ...")
    feature_df = dataset["url"].apply(lambda u: pd.Series(extract_features_14(u)))
    data_final = pd.concat([dataset[["url", "label"]], feature_df], axis=1)

    X = data_final[FEATURE_ORDER]
    y = data_final["label"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
    )
    print("Data latih:", X_train.shape, "| Validasi:", X_val.shape, "| Uji:", X_test.shape)

    param_grid = {"n_estimators": [100, 200], "max_depth": [10, 20]}
    print("Menjalankan Grid Search CV ...")
    t0 = time.time()
    grid = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1),
        param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=1,
    )
    grid.fit(X_train, y_train)
    print(f"Selesai dalam {time.time() - t0:.1f} detik")
    print("Parameter terbaik:", grid.best_params_)

    model = grid.best_estimator_

    from sklearn.metrics import accuracy_score, f1_score
    y_pred = model.predict(X_test)
    print("Accuracy (data uji):", accuracy_score(y_test, y_pred))
    print("F1-score (data uji):", f1_score(y_test, y_pred))

    joblib.dump(model, "model/random_forest_phishing_model.joblib")
    joblib.dump(FEATURE_ORDER, "model/feature_columns.joblib")
    print("Model tersimpan di model/random_forest_phishing_model.joblib")


if __name__ == "__main__":
    main()
