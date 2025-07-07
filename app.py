import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import warnings

# Ignore warnings for cleaner output
warnings.filterwarnings("ignore")

# --- Page Configuration ---
st.set_page_config(
    page_title="Dashboard Uji Model Time Series",
    page_icon="📈",
    layout="wide"
)

# --- Function to load and process data ---
@st.cache_data
def load_data():
    """
    Loads and preprocesses the airline passenger dataset.
    The data is cached to avoid reloading on every interaction.
    """
    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
    df = pd.read_csv(url)
    df.rename(columns={'Month': 'Bulan', 'Passengers': 'Jumlah_Penumpang'}, inplace=True)
    df['Bulan'] = pd.to_datetime(df['Bulan'], format='%Y-%m')
    df.set_index('Bulan', inplace=True)
    return df

# --- Main Application ---
st.title("📈 Dashboard Uji Coba Model Time Series")
st.write("Gunakan dashboard ini untuk menguji performa model **SARIMA** dengan parameter yang berbeda secara interaktif.")

# Load the data
data = load_data()

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header("⚙️ Pengaturan Model")
    
    # Slider for train-test split
    train_split_percentage = st.slider(
        "Persentase Data Latih (%)", 
        min_value=50, 
        max_value=95, 
        value=80, 
        step=5,
        help="Geser untuk menentukan berapa banyak data yang digunakan untuk melatih model."
    )

    st.subheader("Parameter SARIMA (p, d, q)")
    # Input for non-seasonal order
    p = st.number_input("p (Orde AR)", min_value=0, max_value=5, value=1)
    d = st.number_input("d (Orde Diferensiasi)", min_value=0, max_value=5, value=1)
    q = st.number_input("q (Orde MA)", min_value=0, max_value=5, value=1)

    st.subheader("Parameter Musiman (P, D, Q, m)")
    # Input for seasonal order
    P = st.number_input("P (Orde AR Musiman)", min_value=0, max_value=5, value=1)
    D = st.number_input("D (Orde Diferensiasi Musiman)", min_value=0, max_value=5, value=1)
    Q = st.number_input("Q (Orde MA Musiman)", min_value=0, max_value=5, value=1)
    m = st.number_input("m (Periode Musiman)", min_value=1, max_value=24, value=12, help="Biasanya 12 untuk data bulanan.")

    # Button to run the model
    run_button = st.button("Latih dan Prediksi Model", type="primary", use_container_width=True)


# --- Main Panel for Outputs ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Visualisasi Data Penumpang")
    # Display the original data plot
    fig_data, ax_data = plt.subplots(figsize=(12, 6))
    ax_data.plot(data.index, data['Jumlah_Penumpang'], label='Jumlah Penumpang Aktual', color='#0072B2')
    ax_data.set_title('Jumlah Penumpang Maskapai Bulanan (1949-1960)')
    ax_data.set_xlabel('Tahun')
    ax_data.set_ylabel('Jumlah Penumpang')
    ax_data.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig_data)

with col2:
    st.subheader("Data Mentah")
    st.dataframe(data.head())


# --- Model Training and Prediction Logic ---
if run_button:
    st.divider()
    st.header("Hasil Prediksi Model")

    # Split data based on user input
    train_size = int(len(data) * (train_split_percentage / 100))
    train_data, test_data = data['Jumlah_Penumpang'][0:train_size], data['Jumlah_Penumpang'][train_size:]

    # Define model orders from user input
    order = (p, d, q)
    seasonal_order = (P, D, Q, m)

    with st.spinner(f"Melatih model SARIMA dengan order={order} dan seasonal_order={seasonal_order}..."):
        try:
            # Build and fit the SARIMA model
            model_sarima = SARIMAX(train_data, order=order, seasonal_order=seasonal_order)
            fit_sarima = model_sarima.fit(disp=False)

            # Make predictions
            start_index = len(train_data)
            end_index = len(data) - 1
            predictions = fit_sarima.predict(start=start_index, end=end_index, typ='levels')

            # Calculate error
            rmse = np.sqrt(mean_squared_error(test_data, predictions))

            # Display results
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric(label="Root Mean Squared Error (RMSE)", value=f"{rmse:.2f}", help="Semakin rendah nilainya, semakin baik modelnya.")
            
            with res_col2:
                 st.info(f"Model dilatih dengan {train_split_percentage}% data.")

            # Visualize the prediction results
            col3, col4 = st.columns([2, 1])

            with col3:
                st.subheader("Grafik Perbandingan Prediksi")
                fig_pred, ax_pred = plt.subplots(figsize=(12, 6))
                ax_pred.plot(train_data.index, train_data, label='Data Latih', color='gray')
                ax_pred.plot(test_data.index, test_data, label='Data Aktual (Test)', color='#0072B2', marker='o', linestyle='--')
                ax_pred.plot(predictions.index, predictions, label='Prediksi SARIMA', color='#D55E00', marker='x')
                ax_pred.set_title('Perbandingan Model SARIMA dengan Data Aktual', fontsize=16)
                ax_pred.set_xlabel('Tahun', fontsize=12)
                ax_pred.set_ylabel('Jumlah Penumpang', fontsize=12)
                ax_pred.legend()
                ax_pred.grid(True, linestyle='--', alpha=0.6)
                st.pyplot(fig_pred)

            with col4:
                # Show prediction data
                st.subheader("Data Prediksi vs Aktual")
                comparison_df = pd.DataFrame({'Data Aktual': test_data, 'Hasil Prediksi': predictions})
                st.dataframe(comparison_df)

        except Exception as e:
            st.error(f"Terjadi kesalahan saat melatih model: {e}")
            st.warning("Coba gunakan kombinasi parameter (p,d,q) dan (P,D,Q,m) yang berbeda. Kombinasi tertentu dapat menyebabkan error konvergensi.")

