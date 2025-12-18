import requests
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os  # Tambahan untuk manajemen file
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import date, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams

# Set font untuk mendukung karakter Indonesia
rcParams['font.family'] = 'DejaVu Sans'

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Prediksi Harga Ethereum (GRU)",
    page_icon="🪙",
    layout="wide" # Opsional: agar tampilan lebih luas
)

# --- KONFIGURASI FILE ADMIN ---
DATA_FOLDER = 'dataset'
DATA_FILE = 'databackup_eth.csv'
DATA_PATH = os.path.join(DATA_FOLDER, DATA_FILE)

# Pastikan folder dataset ada
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

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

/* --- PERBAIKAN DISINI: MEMBATASI LEBAR KONTEN --- */
/* Ini memaksa konten tetap di tengah dengan lebar maksimal tertentu */
.block-container {
    max-width: 1000px; /* Atur angka ini sesuai selera (misal 900px - 1200px) */
    padding-top: 2rem;
    padding-bottom: 2rem;
    margin: auto; /* Posisi otomatis di tengah */
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


# --- Fungsi-fungsi Bantuan (JANGAN UBAH LOGIKA UTAMA) ---

@st.cache_data(ttl="1h") 
def load_eth_data():
    """
   
    """
    # Cek apakah file admin ada
    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH)
            
            # --- PEMBERSIHAN DATA AGAR SESUAI FORMAT MODEL ---
            # 1. Hapus kolom index lama jika ada
            if "Unnamed: 0" in df.columns:
                df = df.drop(columns=["Unnamed: 0"])
            
            # 2. Standarisasi nama kolom Date
            if "Date" not in df.columns:
                # Coba cari kolom yang mirip 'date'
                found = False
                for col in df.columns:
                    if col.lower() == "date":
                        df = df.rename(columns={col: "Date"})
                        found = True
                        break
                if not found:
                    # Jika tidak ada header Date, asumsikan kolom pertama adalah Date
                    df = df.rename(columns={df.columns[0]: 'Date'})
            
            # 3. Konversi ke datetime
            df["Date"] = pd.to_datetime(df["Date"])
            # Hapus timezone jika ada agar kompatibel dengan matplotlib/numpy
            if df["Date"].dt.tz is not None:
                df["Date"] = df["Date"].dt.tz_localize(None)
                
            return df, "admin_file"
            
        except Exception as e:
            return None, f"error_read: {str(e)}"
    else:
        return None, "no_file"

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
    """Load model & scaler hasil training."""
    model_path = Path("gru_model.h5")
    scaler_path = Path("scaler_gru.pkl")

    if not model_path.exists() or not scaler_path.exists():
        return None, None

    try:
        model = load_model(model_path, compile=False)
        scaler = joblib.load(scaler_path)
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
    """Membuat grafik gabungan."""
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


# --- HALAMAN: DASHBOARD PUBLIK (Landing Page) ---
def show_dashboard():
    st.markdown("<div class='app-title'>Prediksi Harga Ethereum (GRU)</div>", unsafe_allow_html=True)
    st.markdown("<div class='app-subtitle'>Data historis & prediksi ETH-USD berbasis GRU</div>", unsafe_allow_html=True)

    df, data_source = load_eth_data()
    model, scaler = load_gru_assets()

    # LOGIC CHECK DATA
    if data_source == "no_file":
        st.warning("⚠️ Data historis belum tersedia. Silakan hubungi Admin untuk mengupload 'databackup_eth'.")
        return # Stop eksekusi jika data tidak ada
    elif "error" in data_source:
        st.error(f"❌ Terjadi kesalahan membaca data: {data_source}")
        return

    if df is not None and model is not None and scaler is not None:
        # Menampilkan tabel data historis
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='h-section'>📚 Data Historis Harga Ethereum </div>", unsafe_allow_html=True)
        
        # --- PERUBAHAN DISINI: HAPUS .tail(100) ---
        # Menggunakan height=400 agar tabel lebih tinggi dan bisa di-scroll
        st.dataframe(df, height=400, use_container_width=True) 

        # Menampilkan visualisasi line chart
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='h-section'>📈 Visualisasi Harga Historis</div>", unsafe_allow_html=True)
        # Grafik ini sudah otomatis mengambil full data df
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
                        f"Butuh {time_step} hari data sebelumnya."
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

# --- HALAMAN: ADMIN LOGIN & UPLOAD ---
def show_admin_page():
    st.title("🔐 Admin Panel")
    st.write("Area khusus admin untuk memperbarui dataset harga Ethereum.")

    # Simple Password check
    password = st.text_input("Masukkan Password Admin:", type="password")
    
    if password == "admin123": # Ganti password sesuai keinginan
        st.success("Akses Diterima.")
        st.divider()
        
        st.subheader("📂 Upload Dataset Baru")
        st.info(f"File akan disimpan sebagai: `{DATA_FILE}` di dalam folder sistem.")
        
        uploaded_file = st.file_uploader("Upload file CSV (Format: Date, Close, dll)", type=['csv'])
        
        if uploaded_file is not None:
            # Baca preview
            try:
                df_preview = pd.read_csv(uploaded_file)
                st.write("Preview Data:", df_preview.head())
                
                if st.button("💾 Simpan & Update Sistem"):
                    # Simpan file ke path yang ditentukan
                    df_preview.to_csv(DATA_PATH, index=False)
                    
                    # Clear cache agar halaman public langsung berubah
                    st.cache_data.clear()
                    
                    st.success(f"Berhasil! Data telah diperbarui. Pengguna publik sekarang melihat data baru.")
                    
            except Exception as e:
                st.error(f"File error: {e}")
                
    elif password:
        st.error("Password Salah.")


# --- MAIN NAVIGATION (SIDEBAR) ---
# Ini adalah logika utama yang memisahkan tampilan Admin dan User Biasa

st.sidebar.title("Navigasi")
menu = st.sidebar.radio("Pilih Halaman:", ["Dashboard (Public)", "Admin Panel"])

if menu == "Dashboard (Public)":
    show_dashboard()
elif menu == "Admin Panel":
    show_admin_page()