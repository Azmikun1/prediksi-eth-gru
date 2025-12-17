import requests
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import date, timedelta
import yfinance as yf
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams

# Set font untuk mendukung karakter Indonesia
rcParams['font.family'] = 'DejaVu Sans'

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Prediksi Harga Ethereum (GRU)",
    page_icon="🪙"
)

# --- GLOBAL STYLES (UI only) ---
st.markdown("""
<style>
/* font & warna dasar */
html, body, [class*="css"]  { font-family: "Inter", "DejaVu Sans", sans-serif; }
:root {
  --card-bg: #ffffff;
  --muted: #6b7280;
  --ring: #e5e7eb;
}

/* gradient title */
.app-title {
  text-align:center;
  font-weight:800;
  font-size: 32px;
  background: linear-gradient(90deg, #6EE7F9 0%, #7C3AED 50%, #F59E0B 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin: 0.25rem 0 0.5rem 0;
}

/* subtitle */
.app-subtitle {
  text-align:center;
  color: var(--muted);
  margin-bottom: 1.25rem;
}

/* card container */
.card {
  background: var(--card-bg);
  border: 1px solid var(--ring);
  border-radius: 16px;
  padding: 1rem 1.25rem;
  box-shadow: 0 6px 20px rgba(0,0,0,0.05);
  margin-bottom: 1rem;
}

/* section heading */
.h-section {
  font-weight:700;
  font-size: 20px;
  margin: 0 0 .5rem 0;
}

/* table tweaks */
.dataframe tbody tr:hover { background-color: #fafafa; }

/* footer */
.footer {
  text-align:center;
  color: var(--muted);
  font-size: 13px;
  margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)


# --- Fungsi-fungsi Bantuan ---

@st.cache_data(ttl="1h") 
def load_eth_data():
    """
    Metode Hybrid (Tumpuk Data):
    1. Baca Data Lama dari CSV.
    2. Download Data Baru (dari tanggal terakhir CSV sampai Hari Ini).
    3. Gabung (Concat) tanpa mengubah/mengisi data kosong.
    """
    ticker = "ETH-USD"
    df_final = None
    
    # --- BAGIAN 1: BACA DATA LAMA (BASE) ---
    try:
        df_base = pd.read_csv("eth_backup.csv")
        
        # Bersihkan kolom sampah
        if "Unnamed: 0" in df_base.columns:
            df_base = df_base.drop(columns=["Unnamed: 0"])
        
        # Standarisasi kolom Date
        if "Date" not in df_base.columns:
             # Cek kolom pertama
             df_base = df_base.rename(columns={df_base.columns[0]: "Date"})
        
        df_base["Date"] = pd.to_datetime(df_base["Date"]).dt.tz_localize(None)
        
    except FileNotFoundError:
        # Jika tidak ada file, buat dataframe kosong
        df_base = pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])

    # --- BAGIAN 2: DOWNLOAD DATA BARU (INCREMENTAL) ---
    today = date.today()
    
    # Tentukan tanggal mulai download (Lanjutkan dari data terakhir di CSV)
    if not df_base.empty:
        last_date_csv = df_base["Date"].max()
        start_download = last_date_csv + timedelta(days=1)
    else:
        start_download = pd.to_datetime("2024-01-01")

    # Hanya download jika ada selisih hari
    if start_download.date() <= today:
        try:
            # Download dari tanggal terakhir CSV s/d Hari Ini
            df_new = yf.download(
                ticker, 
                start=start_download, 
                end=today + timedelta(days=1), 
                progress=False,
                auto_adjust=True,
                multi_level_index=False 
            )
            
            if df_new is not None and not df_new.empty:
                df_new = df_new.reset_index()
                
                # Rapikan kolom (Hapus MultiIndex jika ada)
                new_cols = []
                for col in df_new.columns:
                    col_name = col[0] if isinstance(col, tuple) else str(col)
                    new_cols.append(col_name)
                df_new.columns = new_cols
                
                # Pastikan kolom Date benar
                if 'Date' not in df_new.columns:
                    df_new = df_new.rename(columns={df_new.columns[0]: 'Date'})
                
                df_new['Date'] = pd.to_datetime(df_new['Date']).dt.tz_localize(None)
                
                # --- BAGIAN 3: GABUNGKAN (CONCAT) ---
                # Tumpuk data lama (Base) dengan data baru (New)
                df_final = pd.concat([df_base, df_new], ignore_index=True)
                
            else:
                # Jika download kosong (misal libur/gagal), pakai data lama saja
                df_final = df_base
                
        except Exception as e:
            print(f"Gagal update online: {e}")
            df_final = df_base # Jika error, tetap tampilkan data lama (Safety Net)
    else:
        # Data CSV sudah paling update
        df_final = df_base

    # --- FINALISASI ---
    if df_final is not None and not df_final.empty:
        # Hapus duplikat (jika ada irisan tanggal)
        df_final = df_final.drop_duplicates(subset="Date", keep="last")
        
        # Urutkan berdasarkan tanggal
        df_final = df_final.sort_values("Date").reset_index(drop=True)
        
        # Filter hanya kolom standar (buang kolom sampah)
        target_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        available = [c for c in target_cols if c in df_final.columns]
        df_final = df_final[available]

        return df_final, "mixed" # Status mixed (Gabungan)

    return None, "error"
    
def validate_scaler(scaler):
    """Validasi scaler agar konsisten dengan training."""
    issues = []
    warnings = []

    if not isinstance(scaler, MinMaxScaler):
        warnings.append(f"Scaler bukan MinMaxScaler (terdeteksi: {type(scaler).__name__}). Pastikan ini scaler training.")

    required_attrs = ["n_features_in_", "data_min_", "data_max_", "scale_", "min_"]
    for a in required_attrs:
        if not hasattr(scaler, a):
            issues.append(f"Scaler belum ter-fit atau tidak valid (atribut '{a}' tidak ada).")

    if hasattr(scaler, "n_features_in_"):
        if int(scaler.n_features_in_) != 1:
            issues.append(f"Scaler mengharapkan {scaler.n_features_in_} fitur, tapi aplikasi hanya pakai 1 fitur ('Close').")

    ok = (len(issues) == 0)
    return ok, issues, warnings


def validate_model_input(model):
    """Pastikan input model bentuknya (None, time_step, 1)."""
    issues = []
    try:
        shape = model.input_shape  # biasanya (None, time_step, 1)
        if not (isinstance(shape, (list, tuple)) and len(shape) == 3):
            issues.append(f"input_shape tidak sesuai harapan: {shape} (harus 3 dimensi).")
        else:
            if shape[2] != 1:
                issues.append(f"Jumlah fitur input model = {shape[2]}, tapi aplikasi membentuk fitur=1.")
            if shape[1] is None or int(shape[1]) <= 1:
                issues.append(f"time_step tidak valid: {shape[1]}.")
    except Exception as e:
        issues.append(f"Gagal membaca input_shape model: {e}")

    ok = (len(issues) == 0)
    return ok, issues


@st.cache_resource
def load_gru_assets():
    """Load model & scaler hasil training (wajib pakai scaler yang sama)."""
    model_path = Path("gru_model.h5")
    scaler_path = Path("scaler_gru.pkl")

    if not model_path.exists() or not scaler_path.exists():
        st.warning(
            f"File tidak ditemukan.\n"
            f"Model: {model_path.resolve().name} ada? {model_path.exists()}\n"
            f"Scaler: {scaler_path.resolve().name} ada? {scaler_path.exists()}"
        )
        return None, None

    try:
        model = load_model(model_path, compile=False)
        scaler = joblib.load(scaler_path)

        ok_scaler, scaler_issues, scaler_warnings = validate_scaler(scaler)
        for w in scaler_warnings:
            st.warning("⚠️ " + w)
        if not ok_scaler:
            st.error("❌ Scaler tidak kompatibel:\n- " + "\n- ".join(scaler_issues))
            return None, None

        ok_model, model_issues = validate_model_input(model)
        if not ok_model:
            st.error("❌ Model input tidak kompatibel:\n- " + "\n- ".join(model_issues))
            return None, None

        return model, scaler

    except Exception as e:
        st.error(f"Gagal load aset: {e}")
        return None, None


def predict_from_sequence_pure(model, initial_sequence_scaled, n_days):
  
    
    seq = np.array(initial_sequence_scaled, dtype="float32").reshape(-1)  # (time_step,)
    preds = []

    for _ in range(n_days):
        x = seq.reshape(1, -1, 1)                    # (1, time_step, 1)
        y = float(model.predict(x, verbose=0)[0, 0]) # output model (scaled)
        preds.append(y)
        seq = np.append(seq[1:], y)                  # autoregressive (murni)

    return np.array(preds, dtype="float32").reshape(-1, 1)


def create_combined_chart(df, start_date, future_dates, future_predictions):
    """Membuat grafik gabungan dengan gaya yang sama seperti referensi."""
    context_start_date = start_date - timedelta(days=60)
    recent_df = df[(df["Date"] >= context_start_date) & (df["Date"] <= start_date)].copy()

    window_size = 7
    recent_df["Trend_Aktual"] = recent_df["Close"].rolling(window=window_size).mean()

    predictions_flat = future_predictions.flatten()

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.plot(
        recent_df["Date"], recent_df["Close"],
        color="gray", linewidth=1.0, label="Harga Aktual (Harian)", alpha=0.5
    )
    ax.plot(
        recent_df["Date"], recent_df["Trend_Aktual"],
        color="#1f77b4", linewidth=2.5, label="Tren Harga Aktual", alpha=0.9
    )
    ax.plot(
        future_dates, predictions_flat,
        color="#d62728", linewidth=2.5, label="Harga Prediksi (GRU)",
        alpha=0.9, marker="o", markersize=4
    )
    ax.scatter(
        recent_df["Date"], recent_df["Close"],
        color="black", s=15, alpha=0.6, zorder=5, label="Data Harian Aktual"
    )

    # garis penghubung trend terakhir ke prediksi pertama
    try:
        last_trend_date = recent_df["Date"].iloc[-1]
        last_trend_price = recent_df["Trend_Aktual"].dropna().iloc[-1]
        first_prediction_date = future_dates[0]
        first_prediction_price = predictions_flat[0]

        ax.plot(
            [last_trend_date, first_prediction_date],
            [last_trend_price, first_prediction_price],
            color="#d62728", linestyle="--", linewidth=2.0, alpha=0.7
        )
    except Exception:
        pass

    ax.set_xlabel("Waktu", fontsize=12, fontweight="bold")
    ax.set_ylabel("Harga (USD)", fontsize=12, fontweight="bold")
    ax.set_title("Analisis Tren Historis dan Prediksi Harga Ethereum", fontsize=16, fontweight="bold", pad=20)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d, %Y"))
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3, linestyle="--")

    handles, labels = ax.get_legend_handles_labels()
    order = [2, 0, 1, 3]
    if len(handles) == 4:
        ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc="upper left", fontsize=11)
    else:
        ax.legend(loc="upper left", fontsize=11)

    plt.tight_layout()
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("white")

    return fig


# --- Tampilan Antarmuka Aplikasi ---

st.markdown("<div class='app-title'>Prediksi Harga Ethereum (GRU)</div>", unsafe_allow_html=True)
st.markdown("<div class='app-subtitle'>Data historis & prediksi ETH-USD berbasis GRU</div>", unsafe_allow_html=True)

df, data_source = load_eth_data()
model, scaler = load_gru_assets()

# LOGIC UI (TOAST/WARNING) DITARUH DI SINI (DILUAR CACHE)
if data_source == "backup":
    st.toast("Koneksi Yahoo lambat. Menggunakan data backup lokal.", icon="⚠️")
elif data_source == "error":
    st.error("❌ Gagal memuat data (Online gagal & Backup tidak ada).")
    
if df is not None and model is not None and scaler is not None:
    # Menampilkan tabel data historis
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='h-section'>📚 Data Historis Harga Ethereum</div>", unsafe_allow_html=True)
    st.dataframe(df, height=300, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Menampilkan visualisasi line chart
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='h-section'>📈 Visualisasi Harga Historis</div>", unsafe_allow_html=True)
    st.line_chart(df.rename(columns={"Date": "index"}).set_index("index")["Close"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='h-section'>🎯 Mulai Prediksi Berdasarkan Tanggal</div>", unsafe_allow_html=True)

    time_step = int(model.input_shape[1])
    min_selectable_date = (df["Date"].min() + timedelta(days=time_step)).date()
    max_selectable_date = df["Date"].max().date()

    col1, col2 = st.columns(2)
    with col1:
        selected_date = st.date_input(
            "Pilih tanggal mulai prediksi:",
            value=max_selectable_date,
            min_value=min_selectable_date,
            max_value=max_selectable_date,
            help=f"Pilih tanggal antara {min_selectable_date} dan {max_selectable_date}"
        )
    with col2:
        prediction_days = st.slider(
            "Pilih jumlah hari untuk prediksi (1-30 hari):",
            min_value=1, max_value=30, value=15, step=1
        )

    if st.button("Buat Prediksi", key="predict_button"):
        if prediction_days < 1:
            st.warning("Jumlah hari prediksi minimal **1**. Silakan geser slidernya dulu. 🙂")
            st.stop()

        with st.spinner("Memproses dan menjalankan prediksi..."):
            # scaling (mengikuti scaler training)
            close_values = df[["Close"]].values.astype(float)
            scaled_data = scaler.transform(close_values)

            # Cocokkan berdasarkan tanggal (robust)
            selected_date_dt = pd.to_datetime(selected_date)
            mask = df["Date"].dt.date == selected_date_dt.date()
            idxs = np.where(mask.to_numpy())[0]

            if len(idxs) == 0:
                st.error(f"Tanggal {selected_date_dt.strftime('%Y-%m-%d')} tidak ditemukan dalam dataset.")
                st.stop()

            start_index = int(idxs[0])

            if start_index - time_step < 0:
                st.error(
                    f"Tanggal terlalu awal untuk diprediksi. "
                    f"Butuh {time_step} hari data sebelumnya. Pilih tanggal setelah "
                    f"{(df['Date'].min() + timedelta(days=time_step)).strftime('%Y-%m-%d')}."
                )
                st.stop()

            initial_sequence = scaled_data[start_index - time_step: start_index]  # (time_step,1)

            # ✅ Prediksi MURNI dari model
            future_predictions_scaled = predict_from_sequence_pure(model, initial_sequence, prediction_days)

            # ✅ Inverse transform MURNI dari scaler
            future_predictions = scaler.inverse_transform(future_predictions_scaled)

            # sanity-check
            if not np.isfinite(future_predictions).all():
                st.error("❌ Hasil prediksi mengandung NaN/Inf. Periksa kompatibilitas model/scaler.")
                st.stop()

            future_dates = [selected_date_dt + timedelta(days=i) for i in range(1, prediction_days + 1)]

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("📊 Grafik Prediksi (Historis + Prediksi)")
            fig = create_combined_chart(df, selected_date_dt, future_dates, future_predictions)
            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("🧾 Tabel Detail Prediksi")
            prediction_table_df = pd.DataFrame({
                "Tanggal": [d.strftime("%Y-%m-%d") for d in future_dates],
                "Harga Prediksi (USD)": [f"USD {price[0]:,.0f}" for price in future_predictions]
            })
            prediction_table_df.index = prediction_table_df.index + 1
            st.dataframe(prediction_table_df, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            last_price = float(df.loc[start_index, "Close"])
            max_prediction = float(np.max(future_predictions))
            min_prediction = float(np.min(future_predictions))

            st.info(f"""
📊 **Informasi Prediksi:**
- Harga terakhir pada tanggal {selected_date.strftime('%Y-%m-%d')}: USD {last_price:,.0f}
- Harga prediksi tertinggi: USD {max_prediction:,.0f}
- Harga prediksi terendah: USD {min_prediction:,.0f}
""")

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='footer'>Dibuat oleh Muhammad Gilman Nadhif Azmi</div>", unsafe_allow_html=True)

elif model is None:
    st.error("Gagal memuat model. Pastikan file 'gru_model.h5' ada di folder yang sama dengan aplikasi.")
elif scaler is None:
    st.error("Gagal memuat scaler. Pastikan file 'scaler_gru.pkl' ada di folder yang sama dengan aplikasi.")
else:
    st.error("Gagal memuat data. Coba cek koneksi internet atau sumber data yfinance.")
