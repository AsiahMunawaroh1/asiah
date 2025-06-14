import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linprog

st.set_page_config(page_title="Aplikasi Model Matematika Industri", layout="wide")

# Sidebar Instructions
st.sidebar.title("📚 Instruksi Penggunaan")
st.sidebar.markdown("""
Aplikasi ini memiliki 4 model matematika untuk keperluan industri:

1. **Optimasi Produksi (Linear Programming):**  
   - Input koefisien fungsi tujuan & kendala.  
   - Hasil berupa solusi optimal & grafik wilayah solusi.

2. **Model Persediaan (EOQ):**  
   - Input permintaan, biaya pesan, biaya simpan & harga satuan.  
   - Hasil: EOQ, total biaya, grafik biaya vs kuantitas.

3. **Model Antrian (M/M/1):**  
   - Input laju kedatangan (λ) & pelayanan (μ).  
   - Hasil: metrik antrian & grafik karakteristik sistem.

4. **Forecasting Permintaan (Simple Exponential Smoothing):**  
   - Input data permintaan historis & alpha.  
   - Hasil: prediksi permintaan & grafik aktual vs prediksi.

**Pilih menu/tab di atas untuk memulai.**
""")

st.title("📈 Aplikasi Model Matematika Industri")

tab1, tab2, tab3, tab4 = st.tabs([
    "Optimasi Produksi (Linear Programming)",
    "Model Persediaan (EOQ)",
    "Model Antrian (M/M/1)",
    "Forecasting Permintaan"
])

with tab1:
    st.header("Optimasi Produksi (Linear Programming)")
    st.markdown("""
    Model ini memaksimalkan/minimumkan fungsi tujuan linear dengan kendala linear.  
    Contoh aplikasi: menentukan jumlah produksi optimal dari 2 produk.
    """)
    st.subheader("Input Data")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Fungsi Tujuan:** (Max) c1*x1 + c2*x2")
        c1 = st.number_input("Koefisien x1 (c1)", value=5.0)
        c2 = st.number_input("Koefisien x2 (c2)", value=4.0)
    with col2:
        st.markdown("**Kendala:**")
        a11 = st.number_input("a11 (x1 di Kendala 1)", value=6.0)
        a12 = st.number_input("a12 (x2 di Kendala 1)", value=4.0)
        b1 = st.number_input("RHS Kendala 1", value=24.0)
        a21 = st.number_input("a21 (x1 di Kendala 2)", value=1.0)
        a22 = st.number_input("a22 (x2 di Kendala 2)", value=2.0)
        b2 = st.number_input("RHS Kendala 2", value=6.0)

    if st.button("Hitung Solusi Optimal (LP)"):
        c = [-c1, -c2]  # linprog minimizes, so multiply by -1 to maximize
        A = [[a11, a12], [a21, a22]]
        b = [b1, b2]
        x_bounds = (0, None)
        res = linprog(c, A_ub=A, b_ub=b, bounds=[x_bounds, x_bounds], method='highs')
        if res.success:
            st.success("Solusi optimal ditemukan!")
            st.write(f"x1 = {res.x[0]:.2f}")
            st.write(f"x2 = {res.x[1]:.2f}")
            st.write(f"Nilai maksimum fungsi tujuan: {(-res.fun):.2f}")

            # Grafik Wilayah Solusi
            st.subheader("Visualisasi Wilayah Solusi")
            x = np.linspace(0, max(b1/a11, b2/a21)*1.2, 200)
            y1 = (b1 - a11*x)/a12
            y2 = (b2 - a21*x)/a22
            plt.figure(figsize=(7,5))
            plt.plot(x, y1, label='Kendala 1')
            plt.plot(x, y2, label='Kendala 2')
            plt.fill_between(x, 0, np.minimum(y1, y2), where=(y1>0) & (y2>0), color='grey', alpha=0.3, label="Wilayah Solusi")
            plt.scatter(res.x[0], res.x[1], color='red', label='Solusi Optimal')
            plt.xlim(left=0)
            plt.ylim(bottom=0)
            plt.xlabel('x1')
            plt.ylabel('x2')
            plt.legend()
            st.pyplot(plt)
        else:
            st.error("Tidak ditemukan solusi optimal, cek kembali input.")

with tab2:
    st.header("Model Persediaan (EOQ)")
    st.markdown("""
    Economic Order Quantity (EOQ) menentukan jumlah pesanan optimal untuk meminimalkan total biaya persediaan.
    """)
    st.subheader("Input Data")
    D = st.number_input("Permintaan per tahun (D)", value=1000.0)
    S = st.number_input("Biaya pemesanan per pesanan (S)", value=50.0)
    H = st.number_input("Biaya simpan per unit per tahun (H)", value=2.0)
    P = st.number_input("Harga per unit (opsional)", value=0.0)
    if st.button("Hitung EOQ"):
        EOQ = np.sqrt((2*D*S)/H)
        N = D/EOQ
        TC = N*S + (EOQ/2)*H
        st.success("Hasil Perhitungan EOQ")
        st.write(f"EOQ (Jumlah Order Optimal): **{EOQ:.2f} unit/order**")
        st.write(f"Frekuensi Order (per tahun): **{N:.2f} kali**")
        st.write(f"Total Biaya Persediaan: **Rp {TC:.2f}**")
        if P > 0:
            st.write(f"Total Biaya (termasuk pembelian): **Rp {TC+D*P:.2f}**")

        # Grafik Biaya vs Kuantitas
        st.subheader("Grafik Total Biaya vs Kuantitas Order")
        Qs = np.linspace(1, EOQ*3, 200)
        total_cost = (D/Qs)*S + (Qs/2)*H
        plt.figure(figsize=(7,4))
        plt.plot(Qs, total_cost, label='Total Biaya')
        plt.axvline(EOQ, color='red', linestyle='--', label='EOQ')
        plt.xlabel("Kuantitas Order (Q)")
        plt.ylabel("Total Biaya")
        plt.legend()
        st.pyplot(plt)

with tab3:
    st.header("Model Antrian (M/M/1)")
    st.markdown("""
    Model antrian M/M/1 digunakan untuk menghitung kinerja sistem antrian dengan 1 server, arrival & service berdistribusi Poisson.
    """)
    st.subheader("Input Data")
    lam = st.number_input("Laju kedatangan rata-rata (λ per waktu)", value=2.0)
    mu = st.number_input("Laju pelayanan rata-rata (μ per waktu)", value=3.0)
    if st.button("Hitung Kinerja Antrian"):
        if lam >= mu:
            st.error("Sistem tidak stabil (λ ≥ μ). Lakukan perbaikan kapasitas!")
        else:
            rho = lam/mu
            L = rho/(1-rho)
            Lq = rho**2/(1-rho)
            W = 1/(mu-lam)
            Wq = rho/(mu-lam)
            st.success("Kinerja Sistem Antrian M/M/1")
            st.write(f"Utilisasi (ρ): **{rho:.2f}**")
            st.write(f"Rata-rata pelanggan dlm sistem (L): **{L:.2f}**")
            st.write(f"Rata-rata pelanggan di antrian (Lq): **{Lq:.2f}**")
            st.write(f"Rata-rata waktu dlm sistem (W): **{W:.2f}**")
            st.write(f"Rata-rata waktu tunggu di antrian (Wq): **{Wq:.2f}**")

            # Grafik Probabilitas n pelanggan dalam sistem
            st.subheader("Probabilitas n pelanggan dalam sistem")
            n = np.arange(0, 15)
            Pn = (1-rho)*rho**n
            plt.figure(figsize=(7,4))
            plt.bar(n, Pn)
            plt.xlabel("Jumlah pelanggan dalam sistem (n)")
            plt.ylabel("Probabilitas P(n)")
            plt.title("Distribusi Probabilitas Jumlah Pelanggan dalam Sistem")
            st.pyplot(plt)

with tab4:
    st.header("Forecasting Permintaan (Simple Exponential Smoothing)")
    st.markdown("""
    Model smoothing eksponensial sederhana untuk prediksi permintaan berbasis data historis.
    """)
    st.subheader("Input Data Historis")
    data_str = st.text_area("Masukkan data permintaan (pisahkan dengan koma)", value="120, 135, 128, 140, 138, 145, 143, 148")
    alpha = st.slider("Alpha (smoothing factor)", 0.01, 1.0, 0.3)
    if st.button("Hitung Forecast"):
        try:
            data = np.array([float(x.strip()) for x in data_str.split(",")])
            forecast = [data[0]]
            for t in range(1, len(data)):
                f = alpha*data[t-1] + (1-alpha)*forecast[-1]
                forecast.append(f)
            next_forecast = alpha*data[-1] + (1-alpha)*forecast[-1]
            st.write(f"Forecast periode berikutnya: **{next_forecast:.2f}**")
            df = pd.DataFrame({'Aktual': data, 'Forecast': forecast})
            st.subheader("Grafik Aktual vs Forecast")
            plt.figure(figsize=(8,4))
            plt.plot(df['Aktual'], marker='o', label='Aktual')
            plt.plot(df['Forecast'], marker='s', label='Forecast')
            plt.legend()
            plt.xlabel('Periode')
            plt.ylabel('Permintaan')
            st.pyplot(plt)
        except:
            st.error("Format data tidak valid. Pastikan hanya angka dipisahkan koma.")

st.markdown("""
---
Aplikasi oleh [AsiahMunawaroh1] - 2025
""")
