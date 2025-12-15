# File: test_koneksi.py

import yfinance as yf
import datetime

print("🚀 Memulai skrip pengujian koneksi yfinance...")

# Gunakan ticker yang umum dan tanggal yang baru
ticker_symbol = "ETH-USD"
start_date = "2024-01-01"

print(f"Mencoba mengunduh data untuk ticker: {ticker_symbol}")
print(f"Tanggal mulai: {start_date}")
print("-" * 30)

try:
    # Kita akan mencoba mengunduh data dengan cara yang paling dasar
    data = yf.download(
        tickers=ticker_symbol,
        start=start_date,
        progress=True  # Kita aktifkan progress bar untuk melihat aktivitas jaringan
    )

    print("-" * 30)
    
    # Periksa hasil unduhan
    if data.empty:
        print("❌ HASIL: GAGAL. yfinance tidak menerima data apa pun.")
        print("Ini mengindikasikan kemungkinan adanya masalah pada jaringan, firewall, atau DNS.")
    else:
        print(f"✅ HASIL: SUKSES! Berhasil menerima {len(data)} baris data.")
        print("Berikut adalah 5 baris pertama data yang diterima:")
        print(data.head())

except Exception as e:
    print(f"🚨 TERJADI ERROR KRITIS SAAT EKSEKUSI 🚨")
    print(f"Detail Error: {e}")
    print("Error ini menunjukkan adanya masalah fundamental saat mencoba menghubungi server Yahoo Finance.")

print("\n🔬 Tes Selesai.")