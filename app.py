import pandas as pd
import streamlit as st

import pickle
from os.path import exists


tab1, tab2, tab3, tab4 = st.tabs(
    ["Dataset", "Preprocessing", "Modelling", "Implementasi"]
)


def load_data(data):
    df = pd.read_csv(data)
    return df


with tab1:
    st.header(
        "Peramalan Harga Saham PT. Gudang Garam Indonesia"
    )

    st.write(
        "Aplikasi ini dibuat untuk digunakan memprediksi harga saham dari PT. Gudang Gara Indonesia, Adanya aplikasi ini membantu para investor untuk melihat harga saham dalam kurun waktu tertentu."
    )
    st.write(
        "Data diambil dari finance.yahoo.com Harga Saham PT. Gudang Garam Indonesia"
    )

    st.write('Kelompok :')
    st.write('1. Chendy Tri Wardani (200411100041)')
    st.write('2. Irham Hamed Ayani (200411100114)')

    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.subheader("Dataset Saham PT Gudang Garam Indonesia")
    
    data=pd.read_csv('csv/GGRM.JK.csv')
    st.write(data)
    # st.write(data)

with tab2:
    st.subheader("Preprocessing Data")
    st.write("Data preprocessing adalah langkah awal yang wajib untuk mempersiapkan model sebelum diproses. Data mentah akan dilakukan preprosessing sesuai dengan kebutuhan agar data lebih mudah  diolah. Inisiatif ini diperlukan karena data mentah seringkali tidak lengkap dan memiliki format yang tidak konsisten. Preprocessing data juga penting untuk menyempurnakan data agar lebih mudah dikomputasi.")
    st.write("Transform univariate time series to supervised learning problem")
    st.write("Tampilkan data yang sudah dipreprocessing")
    st.write("Pilih Form untuk melihat data :")
    # checkbox show data after preprocessing minmax pca
    selectbox = st.selectbox(
        "Pilih Data",
        ("Data Open Saham", "Data Open Saham univariate ke multivariate", "Data Open Saham Preprocessing (MinMax Scaler)", "Data Open Saham Preprocessing (PCA)"),
    )
    show_data = False
    show_data2 = False
    show_data3 = False
    show_data4 = False
    if selectbox == "Data Open Saham":
        show_data = True
    elif selectbox == "Data Open Saham univariate ke multivariate":
        show_data2 = True
    elif selectbox == "Data Open Saham Preprocessing (MinMax Scaler)":
        show_data3 = True
    elif selectbox == "Data Open Saham Preprocessing (PCA)":
        show_data4 = True
    
    
    df=pd.read_csv('csv/gudanggaram_open.csv')
    train_std = pd.read_csv("csv/gudanggaram_timeseries.csv")
    train_norm = pd.read_csv("csv/gudanggaram_timeseries_minmax.csv")
    train_pca = pd.read_csv("csv/gudanggaram_timeseries_pca.csv")
    st.markdown("<br><hr>", unsafe_allow_html=True)
    # show data
    if show_data:
        st.write("Data Data Open Saham")
        st.write(df)
    if show_data2:
        st.write("Data Open Saham univariate ke multivariate")
        st.write(train_std)
    if show_data3:
        st.write("Data Open Saham Preprocessing (MinMax Scaler)")
        st.write(train_norm)
    if show_data4:
        st.write("Data Open Saham Preprocessing (PCA)")
        st.write(train_pca)

with tab3:
    st.subheader("Modelling Data")

    st.write('Setelah melalui proses preprocessing data, langkah berikutnya adalah pembentukan model (Modelling). Silahkan pilih model yang ingin digunakan, kemudian tekan tombol "Modelling" untuk memulai proses modelling.')
    st.write("Model yang digunakan adalah KNN Dengan N Neighbors = 3")
    st.write("Berikut grafik dari tampilan setelah modeling  :")
    st.image("model/output.png")
    st.subheader("Akurasi Model")
    st.write("Berikut adalah akurasi model yang anda pilih:")
    st.write("Mean Absolute Percentage Error: ", round(100 * 0.0176, 2), "%")

with tab4:
    st.header("Implementasi Aplikasi")
    st.write(
        'Silahkan isi input dibawah ini dengan benar. Setelah itu tekan tombol "Prediksi" untuk memprediksi'
    )

    if exists('scaler.sav'):
        scaler = pickle.load(open('scaler.sav', 'rb'))
    if exists('pca.sav'):
        pca = pickle.load(open('pca.sav', 'rb'))

    with st.form(key="Form4"):
        input1 = st.number_input("Masukkan Harga Saham (Close) Hari Ini", min_value=0)
        input2 = st.number_input("Masukkan Harga Saham (Close) Kemarin", min_value=0)
        input3 = st.number_input("Masukkan Harga Saham (Close) Kemarin Lusa", min_value=0)

        int1 = int(input1)
        int2 = int(input2)
        int3 = int(input3)

        submitted3 = st.form_submit_button(label="Prediksi")

    if submitted3:
        input = [[int3, int2, int1]]
        # input_norm = scaler.transform(input)
        # input_pca = pca.transform(input_norm)

        knn = pickle.load(open('model/knn.pkl', 'rb'))
        minmax_sca = pickle.load(open('model/minmax_sca.pkl', 'rb'))
        pca = pickle.load(open('model/pca.pkl', 'rb'))

        input_mm = minmax_sca.transform(input)
        input_pca = pca.transform(input_mm)

        # predict new data
        y_pred=knn.predict(input_pca)

        st.write("Prediksi Harga Saham (Close) Besok: ", y_pred[0])