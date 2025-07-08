import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import warnings

# Abaikan warnings untuk output yang lebih bersih
warnings.filterwarnings("ignore")

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Aplikasi Prediksi Time Series",
    page_icon="🔮",
    layout="wide"
)


# --- Fungsi Pemuatan Data dan Model ---

@st.cache_data
def load_data():
    """
    Memuat dan memproses dataset penumpang maskapai.
    """
    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
    df = pd.read_csv(url)
    df.rename(columns={'Month': 'Bulan', 'Passengers': 'Jumlah_Penumpang'}, inplace=True)
    df['Bulan'] = pd.to_datetime(df['Bulan'], format='%Y-%m')
    df.set_index('Bulan', inplace=True)
    return df

@st.cache_resource
def load_model():
    """
    Memuat model SARIMA yang sudah dilatih dari file model/sarima_model.pkl.
    """
    model_path = 'model/sarima_model.pkl'
    if not os.path.exists(model_path):
        st.error(f"File model '{model_path}' tidak ditemukan.")
        st.info("Pastikan file 'model_sarima.pkl' berada di folder yang sama dengan script aplikasi ini.")
        return None
    
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

# --- Aplikasi Utama ---
st.title("🔮 Aplikasi Prediksi Penumpang Maskapai")
st.write("Aplikasi ini menggunakan model SARIMA yang sudah dilatih untuk memprediksi jumlah penumpang di masa depan.")

# Muat data dan model yang sudah dilatih dari file
data = load_data()
model = load_model()

# --- Sidebar untuk Input Pengguna ---
with st.sidebar:
    st.header("⚙️ Pengaturan Prediksi")
    
    # Input untuk jumlah bulan yang akan diprediksi
    n_forecast = st.number_input(
        "Jumlah bulan untuk diprediksi:", 
        min_value=1, 
        max_value=48, 
        value=12, 
        step=1,
        help="Masukkan berapa bulan ke depan Anda ingin melihat prediksi."
    )

    # Tombol untuk menjalankan prediksi
    run_forecast = st.button("Prediksi", type="primary", use_container_width=True)


# --- Panel Utama untuk Output ---
# Hanya tampilkan output jika model berhasil dimuat
if model:
    # Tampilkan data historis terlebih dahulu
    st.subheader("Data Historis Jumlah Penumpang (1949-1960)")
    fig_data, ax_data = plt.subplots(figsize=(12, 6))
    ax_data.plot(data.index, data['Jumlah_Penumpang'], label='Data Historis', color='#0072B2')
    ax_data.set_title('Jumlah Penumpang Maskapai Bulanan')
    ax_data.set_xlabel('Tahun')
    ax_data.set_ylabel('Jumlah Penumpang')
    ax_data.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig_data)

    # --- Logika dan Tampilan Prediksi ---
    if run_forecast:
        st.divider()
        st.header(f"Hasil Prediksi untuk {n_forecast} Bulan ke Depan")

        with st.spinner("Membuat prediksi..."):
            # Gunakan metode .get_forecast() untuk membuat prediksi masa depan
            forecast_result = model.get_forecast(steps=n_forecast)
            
            # Dapatkan hasil prediksi dan interval kepercayaan
            forecast_values = forecast_result.predicted_mean
            confidence_intervals = forecast_result.conf_int()

            # --- Tampilkan hasil dalam layout 2 kolom ---
            res_col1, res_col2 = st.columns([2, 1])

            with res_col1:
                # Visualisasi hasil prediksi
                st.subheader("Grafik Prediksi")
                fig_pred, ax_pred = plt.subplots(figsize=(12, 6))
                
                ax_pred.plot(data.index, data['Jumlah_Penumpang'], label='Data Historis', color='#0072B2')
                ax_pred.plot(forecast_values.index, forecast_values, label='Data Prediksi', color='#D55E00', linestyle='--')
                ax_pred.fill_between(confidence_intervals.index,
                                     confidence_intervals.iloc[:, 0],
                                     confidence_intervals.iloc[:, 1], color='orange', alpha=0.2, label='Interval Kepercayaan (95%)')

                ax_pred.set_title('Prediksi Jumlah Penumpang di Masa Depan')
                ax_pred.set_xlabel('Tahun')
                ax_pred.set_ylabel('Jumlah Penumpang')
                ax_pred.legend()
                ax_pred.grid(True, linestyle='--', alpha=0.6)
                st.pyplot(fig_pred)

            with res_col2:
                st.subheader("Tabel Data Prediksi")
                forecast_df = pd.DataFrame({
                    'Bulan': forecast_values.index.strftime('%Y-%m'),
                    'Prediksi Penumpang': forecast_values.values.astype(int)
                })
                st.dataframe(forecast_df, use_container_width=True)
                st.info("Interval kepercayaan menunjukkan rentang di mana nilai aktual kemungkinan besar akan berada.")