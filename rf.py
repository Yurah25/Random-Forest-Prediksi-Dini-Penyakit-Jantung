from ucimlrepo import fetch_ucirepo
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree 
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
)
import joblib, os, warnings

warnings.filterwarnings("ignore")

OUTPUT_DIR = "outputs_ml"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_DEPTH = 5


print("Sedang mengunduh data langsung dari UCI Repository (ID=45)...")


heart_disease = fetch_ucirepo(id=45) 


X = heart_disease.data.features
y = heart_disease.data.targets

print(f" -> Data berhasil diunduh! Ukuran: {X.shape}")

y = (y > 0).astype(int).values.ravel()
y = pd.Series(y, name="target")

print(f" -> Target berhasil dikonversi ke Biner (0/1).")


num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

X = X[num_cols + cat_cols]

print("\n[INFO] Mengecek dan menghapus data duplikat...")

data_gabungan = pd.concat([X, y], axis=1)
jumlah_awal = len(data_gabungan)

data_gabungan.drop_duplicates(inplace=True)

jumlah_akhir = len(data_gabungan)
jumlah_dihapus = jumlah_awal - jumlah_akhir

if jumlah_dihapus > 0:
    print(f" -> DUPLIKAT DITEMUKAN: {jumlah_dihapus} baris data duplikat telah dihapus.")
else:
    print(" -> AMAN: Tidak ada data duplikat di dataset ini.")

print(f" -> Jumlah data final setelah cek duplikat: {jumlah_akhir} baris.")

X = data_gabungan.drop(columns=["target"])
y = data_gabungan["target"]

print("="*50 + "\n")

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
])
preproc = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])

print("\n[INFO] Sedang memproses penyimpanan data bersih...")
X_cleaned_array = preproc.fit_transform(X)

try:
    new_columns = preproc.get_feature_names_out()
except AttributeError:
    new_columns = [f"feature_{i}" for i in range(X_cleaned_array.shape[1])]

X_cleaned_df = pd.DataFrame(X_cleaned_array, columns=new_columns)

data_cleaned_final = pd.concat([X_cleaned_df, y.reset_index(drop=True)], axis=1)


cleaned_filename = f"{OUTPUT_DIR}/heart_disease_uci_cleaned_final.csv"
data_cleaned_final.to_csv(cleaned_filename, index=False)

print(f" -> SUKSES: Dataset bersih tersimpan di: {cleaned_filename}")
print(f" -> Dimensi Data Final: {data_cleaned_final.shape}")
print("="*50 + "\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"Train: {len(X_train)} data | Test: {len(X_test)} data")

model = Pipeline([
    ("preproc", preproc),
    ("clf", RandomForestClassifier(
        criterion="entropy",       
        n_estimators=100,          
        max_depth=MAX_DEPTH,        
        random_state=RANDOM_STATE,
        n_jobs=-1                    
    ))
])

print("\nTraining Random Forest...")
model.fit(X_train, y_train)

print("\n" + "="*40)
print(" ANALISIS PERFORMA (TRAINING vs TESTING)")
print("="*40)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

acc_train = accuracy_score(y_train, y_pred_train)
acc_test = accuracy_score(y_test, y_pred_test)
recall_test = recall_score(y_test, y_pred_test)

print(f"Akurasi Training : {acc_train:.4f} ({(acc_train*100):.2f}%)")
print(f"Akurasi Testing  : {acc_test:.4f}  ({(acc_test*100):.2f}%)")
print(f"Recall (Sensitivitas): {recall_test:.4f} (Kemampuan mendeteksi orang sakit)")

if (acc_train - acc_test) > 0.10:
    print("-> STATUS: WARNING (Overfitting > 10%)")
else:
    print("-> STATUS: AMAN (Good Fit)")


try:
 
    feature_names = model.named_steps['preproc'].get_feature_names_out()
   
    importances = model.named_steps['clf'].feature_importances_

    df_imp = pd.DataFrame({"Fitur": feature_names, "Pentingnya": importances})
    df_imp = df_imp.sort_values("Pentingnya", ascending=False).head(10) # Ambil Top 10

    print("\n" + "="*40)
    print(" TOP 5 FAKTOR PENYEBAB (MENURUT MODEL)")
    print("="*40)
    print(df_imp.head(5).to_string(index=False))

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Pentingnya", y="Fitur", data=df_imp, palette="viridis")
    plt.title("Top 10 Fitur Paling Berpengaruh (Feature Importance)")
    plt.xlabel("Tingkat Kepentingan")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/feature_importance.png")
    print(f"\n[INFO] Grafik Feature Importance tersimpan di: {OUTPUT_DIR}/feature_importance.png")
    plt.close()

except Exception as e:
    print(f"\n[SKIP] Gagal membuat feature importance: {e}")

print("="*40 + "\n")

joblib.dump(model, f"{OUTPUT_DIR}/random_forest_model.pkl")

print("\nModel dan hasil tersimpan di folder:", OUTPUT_DIR)

    
def interactive_rf_predict(model_pipeline, raw_X_for_examples, numeric_columns, categorical_columns):
    print("\n=== PREDIKSI INTERAKTIF (Random Forest) ===")
    
    sample = raw_X_for_examples.iloc[0] 
    row = {}

    print("Silakan masukkan data (Tekan Enter untuk memakai nilai contoh):")

    print("\n--- Kolom Numerik (Angka) ---")
    for c in numeric_columns:
        v = input(f"  {c} (contoh: {sample[c]}): ").strip()
        
        if v == "":
            row[c] = sample[c]
        else:
            try:
                row[c] = float(v)
            except ValueError:
                print(f"   -> Input tidak valid, menggunakan nilai contoh: {sample[c]}")
                row[c] = sample[c]

    print("\n--- Kolom Kategorikal (Teks) ---")
    for c in categorical_columns:
        v = input(f"  {c} (contoh: {sample[c]}): ").strip()
        
        if v == "":
            row[c] = sample[c]
        else:
            row[c] = v
 
    df_in = pd.DataFrame([row])
    
    pred = model_pipeline.predict(df_in)[0]
    prob = model_pipeline.predict_proba(df_in)[0]

    print("\n" + "="*30)
    print("      HASIL PREDIKSI")
    print("="*30)
    print(f"  Prediksi Target: {int(pred)}")
 
    if int(pred) == 1:
        print(f"  Keyakinan (Probabilitas Penyakit): {prob[1]:.4f} (atau {prob[1]*100:.2f}%)")
    else:
        print(f"  Keyakinan (Probabilitas Sehat): {prob[0]:.4f} (atau {prob[0]*100:.2f}%)")
    print("="*30)


try:
    interactive_rf_predict(model, X, num_cols, cat_cols)
except NameError as e:
    print(f"\nERROR: Gagal menjalankan mode interaktif.")
    print(f"Variabel '{e.name}' tidak ditemukan di program utama Anda.")
except Exception as e:
     print(f"\nERROR tidak terduga di mode interaktif: {e}")