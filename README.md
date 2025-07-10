# Prediksi Jumlah Penumpang Maskapai Menggunakan Model Time Series

## ***Business Understanding***
Dalam industri penerbangan yang sangat kompetitif, kemampuan untuk memprediksi jumlah penumpang di masa depan adalah aset yang krusial. Prediksi yang akurat memungkinkan maskapai untuk mengoptimalkan berbagai aspek operasional dan strategis, mulai dari alokasi sumber daya seperti pesawat dan kru, perencanaan rute penerbangan, hingga manajemen pendapatan dan strategi penetapan harga. Proyek ini bertujuan untuk membangun sebuah model peramalan deret waktu (*time-series forecasting*) yang andal untuk memprediksi jumlah penumpang maskapai bulanan.

## **Permasalahan Bisnis**

Permasalahan bisnis utama yang ingin diselesaikan adalah bagaimana cara membuat peramalan jumlah penumpang bulanan yang akurat berdasarkan data historis. Maskapai memerlukan model yang tidak hanya dapat menangkap tren pertumbuhan jangka panjang tetapi juga pola musiman yang berulang setiap tahun (misalnya, lonjakan penumpang selama musim liburan). Tanpa model peramalan yang baik, maskapai berisiko mengalami kerugian akibat alokasi sumber daya yang tidak efisien, seperti kursi kosong pada penerbangan atau sebaliknya, kekurangan kapasitas saat permintaan tinggi.

## **Cakupan Proyek**

Proyek ini mencakup keseluruhan alur kerja pengembangan model *time-series*, mulai dari persiapan data, pembangunan model, pelatihan, evaluasi, hingga implementasinya dalam sebuah aplikasi web interaktif.

### Pemahaman Data (*Data Understanding*)
Data yang digunakan adalah dataset "Air Passengers" yang berisi data bulanan jumlah penumpang maskapai dari Januari 1949 hingga Desember 1960. Dataset ini terdiri dari dua kolom: '*Month*' dan '*Passengers*'. Dari hasil eksplorasi data, ditemukan beberapa karakteristik kunci:

- Jumlah penumpang secara konsisten meningkat dari tahun ke tahun.

- Terdapat pola yang jelas dan berulang setiap 12 bulan, dengan puncak pada pertengahan tahun (musim panas) dan titik terendah pada awal/akhir tahun.

- Amplitudo atau besarnya fluktuasi musiman juga tampak meningkat seiring berjalannya waktu.

### Persiapan Data (*Data Preparation*)
Data mentah tidak dapat langsung digunakan untuk model ARIMA dan variannya karena adanya tren dan musiman (data tidak stasioner). Langkah-langkah persiapan data yang dilakukan adalah:

- Kolom `'Months'` diubah menjadi format datetime dan dijadikan sebagai indeks data.

- `Augmented Dickey-Fuller (ADF)` test dilakukan dan mengkonfirmasi bahwa data asli tidak stasioner `(p-value > 0.05)`.

- Untuk membuat data menjadi stasioner, dilakukan dua kali diferensiasi:
    - Diferensiasi pertama (`diff(1)`) untuk menghilangkan tren.
    - Diferensiasi musiman (`diff(12)`) untuk menghilangkan pola musiman tahunan.

- Setelah diferensiasi, ADF test menunjukkan bahwa data telah stasioner **(p-value < 0.05)** dan siap untuk dimodelkan.

### *Machine Learning Modeling*
Tahapan pemodelan melibatkan pembangunan tiga model berbeda untuk perbandingan:
- **Model ARIMA (Autoregressive Integrated Moving Average)**: Dibangun sebagai model dasar. Model ini hanya menggunakan parameter non-musiman (p,d,q). Hasilnya, model ini mampu menangkap tren umum tetapi gagal total dalam memodelkan pola musiman.

- **Model SARIMA (Seasonal ARIMA)**: Merupakan pengembangan dari ARIMA dengan menambahkan parameter musiman (P,D,Q,m). Model ini secara eksplisit dirancang untuk data dengan pola musiman.

- **Model SARIMAX (Seasonal ARIMA with eXogenous variables)**: Varian dari SARIMA yang memungkinkan penambahan variabel eksternal. Dalam proyek ini, sebuah variabel tren sederhana ditambahkan untuk demonstrasi.

Parameter untuk model SARIMA dan SARIMAX, yaitu `order=(1, 1, 1)` dan `seasonal_order=(1, 1, 1, 12)`, ditentukan berdasarkan analisis plot `ACF (Autocorrelation Function)` dan `PACF (Partial Autocorrelation Function)` dari data yang telah didiferensiasi.

### *Evaluation*
Evaluasi model dilakukan dengan membagi data menjadi 80% data latih dan 20% data uji. Performa model diukur dengan menghitung `Root Mean Squared Error (RMSE)` antara nilai prediksi pada data uji dan nilai aktualnya.

### *Deployment*
Model terbaik, yaitu SARIMA dengan parameter `order=(1, 1, 1)` dan `seasonal_order=(1, 1, 1, 12)`, dilatih kembali menggunakan seluruh dataset untuk memaksimalkan pembelajaran. Model yang sudah final ini kemudian disimpan ke dalam sebuah file bernama sarima_model.pkl.

Model yang telah disimpan ini kemudian di-deploy melalui Streamlit. Aplikasi ini memuat model dan menyediakan interface bagi pengguna untuk memasukkan jumlah bulan yang ingin diprediksi ke depan.


## **Persiapan**

Sumber data pelatihan: 
Dataset "Air Passengers" via Github
```
https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv
```

*Setup environment*:
```
// virtual enviroment setup
python -m venv .env --> membuat virtual enviroment
.env\Scripts\activate --> mengaktifkan virtual enviroment
pip install -r requirements.txt --> instal requirements

// additional commad
pip list --> melihat library yang terinstal
deactivate --> mematikan virtual enviroment
Remove-Item -Recurse -Force .\.env --> menghapus virtual enviroment
```


## **Menjalankan Sistem *Machine Learning***

Aplikasi ini dibangun dengan Streamlit dan dapat dijalankan secara lokal. Cara menjalankan Aplikasinya adalah sebagai berikut:
1. Pastikan library yang diperlukan telah terinstall (`reqirements.txt`).

2. Pastikan file model sudah didalam direktori.

3. Buka terminal atau command prompt, arahkan ke direktori proyek.

4. Jalankan perintah berikut:
    ```
    streamlit run app.py
    ```

5. Aplikasi akan otomatis terbuka di browser.

6. Gunakan sidebar dan masukkan jumlah bulan yang ingin diprediksi, lalu klik tombol "Prekdiksi" untuk melihat hasilnya.


## ***Conclusion***

Untuk data deret waktu dengan pola musiman yang kuat seperti jumlah penumpang maskapai, penggunaan model yang dapat menangani musiman seperti SARIMA adalah suatu keharusan dan memberikan peningkatan performa yang drastis dibandingkan model ARIMA biasa.