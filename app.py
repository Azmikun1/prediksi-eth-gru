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

# Set font
rcParams['font.family'] = 'DejaVu Sans'

# --- Konfigurasi Halaman ---
st.set_page_config(page_title="Prediksi Harga Ethereum (GRU)", page_icon="🪙")

# --- GLOBAL STYLES ---
st.markdown("""
<style>
.app-title {
  text-align:center; font-weight:800; font-size: 32px;
  background: linear-gradient(90deg, #6EE7F9 0%, #7C3AED 50%, #F59E0B 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  margin: 0.25rem 0 0.5rem 0;
}
.app-subtitle { text-align:center; color: #6b7280; margin-bottom: 1.25rem; }
.card {
  background: #ffffff; border: 1px solid #e5e7eb; border-radius: 16px;
  padding: 1rem 1.25rem; box-shadow: 0 6px 20px rgba(0,0,0,0.05); margin-bottom: 1rem;
}
.h-section { font-weight:700; font-size: 20px; margin: 0 0 .5rem 0; }
.footer { text-align:center; color: #6b7280; font-size: 13px; margin-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# --- TOMBOL RESET CACHE (PENTING!) ---
with st.sidebar:
    st.header("⚙️ Kontrol Data")
    if st.button("🔄 Paksa Update Data (Clear Cache)"):
        st.cache_data.clear()
        st.success("Cache dihapus! Silakan tekan 'R' untuk reload.")
        st.stop() # Hentikan app sebentar biar user reload

# --- FUNGSI LOAD DATA (VERSI YFINANCE ONLY) ---
@st.cache_data(ttl="1h") 
def load_eth_data():
    """
    Fokus: Download via Library yfinance.
    Fitur: 
    1. Mengatasi data bolong (Resample Daily).
    2. Auto Adjust harga (OHLC bersih).
    """
    ticker = "ETH-USD"
    
    # Kita set start date agak jauh biar grafiknya bagus
    start_date = "2024-01-01"
    end_date = date.today() + timedelta(days=1)
    
    st.toast("Sedang menghubungi server Yahoo Finance...", icon="⏳")
    
    try:
        # DOWNLOAD ONLINE
        df = yf.download(
            ticker, 
            start=start_date, 
            end=end_date, 
            progress=False,
            auto_adjust=True, # Biar harga bersih
            multi_level_index=False 
        )
        
        if df is not None and not df.empty:
            # 1. Bersihkan Index
            df = df.reset_index()
            
            # 2. Rapikan Kolom (Cegah MultiIndex/Tuple)
            new_cols = []
            for col in df.columns:
                col_name = col[0] if isinstance(col, tuple) else str(col)
                new_cols.append(col_name)
            df.columns = new_cols
            
            # 3. Pastikan kolom Date ada
            if 'Date' not in df.columns:
                df = df.rename(columns={df.columns[0]: 'Date'})
            
            # 4. Hapus Timezone
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
            
            # --- BAGIAN PENTING: TAMBAL DATA BOLONG (RESAMPLING) ---
            # Ini mengatasi masalah "Loncat" dari tgl 15 ke 17.
            # Kita paksa buat tanggal harian (Daily) lengkap.
            df = df.sort_values('Date').set_index('Date')
            
            # Buat range tanggal penuh dari awal sampai akhir data
            full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
            
            # Reindex & Forward Fill (Isi kekosongan dengan data hari sebelumnya)
            df = df.reindex(full_idx).ffill().reset_index()
            df = df.rename(columns={'index': 'Date'})
            # -------------------------------------------------------

            # 5. Simpan Backup Otomatis
            try:
                df.to_csv("eth_backup.csv", index=False)
            except:
                pass
            
            return df, "online"
            
    except Exception as e:
        print(f"Error yfinance: {e}")

    # FALLBACK: JIKA DOWNLOAD GAGAL, BACA BACKUP LAMA
    try:
        df_backup = pd.read_csv("eth_backup.csv")
        if "Unnamed: 0" in df_backup.columns: df_backup = df_backup.drop(columns=["Unnamed: 0"])
        if "Date" not in df_backup.columns: df_backup = df_backup.rename(columns={df_backup.columns[0]: "Date"})
        df_backup["Date"] = pd.to_datetime(df_backup["Date"]).dt.tz_localize(None)
        
        return df_backup, "backup"
    except:
        return None, "error"


# --- Fungsi Helper Lainnya (Model & Scaler) ---
def validate_scaler(scaler):
    issues, warnings = [], []
    if not isinstance(scaler, MinMaxScaler): warnings.append(f"Scaler bukan MinMaxScaler.")
    return (len(issues)==0), issues, warnings

def validate_model_input(model):
    issues = []
    try:
        shape = model.input_shape
        if shape[2] != 1: issues.append(f"Fitur model = {shape[2]}, input = 1.")
    except: pass
    return (len(issues)==0), issues

@st.cache_resource
def load_gru_assets():
    model_path = Path("gru_model.h5")
    scaler_path = Path("scaler_gru.pkl")
    if not model_path.exists() or not scaler_path.exists(): return None, None
    try:
        model = load_model(model_path, compile=False)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except: return None, None

def predict_from_sequence_pure(model, initial_sequence_scaled, n_days):
    seq = np.array(initial_sequence_scaled, dtype="float32").reshape(-1)
    preds = []
    for _ in range(n_days):
        x = seq.reshape(1, -1, 1)
        y = float(model.predict(x, verbose=0)[0, 0])
        preds.append(y)
        seq = np.append(seq[1:], y)
    return np.array(preds, dtype="float32").reshape(-1, 1)

def create_combined_chart(df, start_date, future_dates, future_predictions):
    context_start = start_date - timedelta(days=60)
    recent_df = df[(df["Date"] >= context_start) & (df["Date"] <= start_date)].copy()
    recent_df["Trend"] = recent_df["Close"].rolling(window=7).mean()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(recent_df["Date"], recent_df["Close"], color="gray", alpha=0.5, label="Harga Aktual")
    ax.plot(recent_df["Date"], recent_df["Trend"], color="#1f77b4", linewidth=2, label="Tren Aktual")
    ax.plot(future_dates, future_predictions.flatten(), color="#d62728", linewidth=2, marker="o", markersize=4, label="Prediksi GRU")
    
    if not recent_df.empty:
        ax.plot([recent_df["Date"].iloc[-1], future_dates[0]], 
                [recent_df["Trend"].dropna().iloc[-1], future_predictions.flatten()[0]], 
                color="#d62728", linestyle="--", alpha=0.7)

    ax.set_title("Analisis Tren & Prediksi Ethereum")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d, %Y"))
    plt.xticks(rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


# --- MAIN UI ---

st.markdown("<div class='app-title'>Prediksi Harga Ethereum (GRU)</div>", unsafe_allow_html=True)
st.markdown("<div class='app-subtitle'>Data historis & prediksi ETH-USD berbasis GRU</div>", unsafe_allow_html=True)

# LOAD DATA
df, status = load_eth_data()

# Notifikasi Status
if status == "backup":
    st.toast("Koneksi Yahoo Gagal. Pakai Backup.", icon="⚠️")
elif status == "error":
    st.error("❌ Gagal memuat data (Library gagal & Backup tidak ada).")
    st.stop()

# LOAD MODEL
model, scaler = load_gru_assets()

if df is not None and model is not None and scaler is not None:
    
    # INFO UPDATE DATA
    last_date = df["Date"].max()
    st.info(f"📅 Data Terupdate sampai: **{last_date.strftime('%d %B %Y')}**")

    # 1. Tampilkan Data
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='h-section'>📚 Data Historis Harga Ethereum</div>", unsafe_allow_html=True)
    st.dataframe(df.tail(10).sort_values("Date", ascending=False), height=300, use_container_width=True) # Tampilkan 10 data terakhir biar kelihatan update
    st.markdown("</div>", unsafe_allow_html=True)

    # 2. Chart Historis
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='h-section'>📈 Visualisasi Harga Historis</div>", unsafe_allow_html=True)
    st.line_chart(df.set_index("Date")["Close"])
    st.markdown("</div>", unsafe_allow_html=True)

    # 3. Prediksi
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='h-section'>🎯 Mulai Prediksi</div>", unsafe_allow_html=True)
    
    time_step = int(model.input_shape[1])
    min_date = (df["Date"].min() + timedelta(days=time_step)).date()
    max_date = df["Date"].max().date()

    c1, c2 = st.columns(2)
    with c1:
        # Default value ke hari ini (max_date)
        start_date = st.date_input("Mulai tanggal:", value=max_date, min_value=min_date, max_value=max_date)
    with c2:
        days = st.slider("Jumlah hari:", 1, 30, 15)

    if st.button("Buat Prediksi"):
        with st.spinner("Memproses..."):
            vals = df[["Close"]].values.astype(float)
            scaled = scaler.transform(vals)
            
            check_date = pd.to_datetime(start_date)
            # Cari index tanggal
            idx = df[df["Date"].dt.date == check_date.date()].index
            
            if len(idx) > 0:
                idx = idx[0]
                if idx < time_step:
                     st.error(f"Data tidak cukup (butuh {time_step} hari sebelumnya).")
                else:
                    seq = scaled[idx-time_step : idx]
                    pred_scaled = predict_from_sequence_pure(model, seq, days)
                    pred_real = scaler.inverse_transform(pred_scaled)
                    
                    f_dates = [check_date + timedelta(days=i) for i in range(1, days+1)]
                    
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    fig = create_combined_chart(df, check_date, f_dates, pred_real)
                    st.pyplot(fig)
                    
                    res_df = pd.DataFrame({"Tanggal": [d.strftime("%Y-%m-%d") for d in f_dates], "Harga (USD)": pred_real.flatten()})
                    st.dataframe(res_df, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.error("Tanggal tidak ditemukan dalam dataset.")

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='footer'>Dibuat oleh Muhammad Gilman Nadhif Azmi</div>", unsafe_allow_html=True)

elif model is None:
    st.error("Gagal memuat Model/Scaler. Pastikan file .h5 dan .pkl ada.")