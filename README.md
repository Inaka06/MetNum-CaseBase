# CB03 Metode Numerik

Perbandingan interpolasi Lagrange dan Natural Cubic Spline pada RSSI nyata UJIIndoorLoc, dengan uji fungsi Runge dan kontur spasial.

## Isi

- `cb3.ipynb`: notebook dengan 26 cell, penjelasan, kode, dan output eksekusi.
- `interp.py`: versi skrip Python yang setara.
- `Laporan_CB03.docx`: laporan analisis dan hasil pengujian.
- `data/ujiindoorloc.zip`: arsip dataset asli UCI, mencakup training dan validation. Kode hanya membaca training.
- `hasil/`: tabel CSV, grafik PNG, serta ringkasan JSON hasil eksekusi.
- `requirements.txt`: dependensi analisis.

## Menjalankan

Disarankan Python 3.11 atau 3.12 dalam virtual environment.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python interp.py
```

Untuk Jupyter, pasang `jupyterlab` dan `ipykernel`, lalu buka `cb3.ipynb` dari folder CB03 dan pilih Run All. Di Colab, notebook dapat mengunduh dataset dari UCI jika arsip lokal tidak ada. Tidak memerlukan kredensial Kaggle. Koneksi internet tidak diperlukan untuk membaca data jika menggunakan paket lengkap ini.

## Ringkasan rancangan

Gedung 2 lantai 3 dipilih karena baris pengukurannya terbanyak. Geometri menentukan pita penampang dan satu koordinat wakil dari setiap kelompok posisi berdekatan. AP dipilih berdasarkan ketersediaan sinyal pada titik wakil, dengan nama sebagai pemecah seri. Median RSSI dihitung hanya di koordinat yang sama. Tidak ada data RSSI sintetis yang menggantikan pengukuran nyata.

Penampang menghasilkan 11 posisi: delapan simpul dan tiga lokasi uji interior. Nilai 100 berarti sinyal tidak terdeteksi dan dikeluarkan sebelum agregasi. Posisi sepanjang garis adalah selisih koordinat proyeksi, bukan jarak fisik pasti dari gateway.

Pada eksekusi yang disertakan, RMSE Lagrange 1.5967 dB, natural spline 0.7466 dB, dan linear 0.3921 dB. Sampel uji sangat kecil, sehingga hasil tidak digeneralisasi untuk semua AP atau jaringan IoT.

Kontur 2D memakai interpolasi linear per segitiga. Pemeriksaan tambahan pada kubik 2D juga dicatat karena menghasilkan perkiraan di luar rentang pengukuran. Ini berbeda dari natural spline 1D. Fenomena Runge klasik diuji terpisah dengan fungsi Runge yang dihitung langsung dari rumus.

## Atribusi dataset

Torres-Sospedra, J., Montoliu, R., Martínez-Usó, A., Arnau, T., dan Avariento, J. (2014). UJIIndoorLoc. UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C5MS59.

Sumber asli: https://archive.ics.uci.edu/dataset/310/ujiindoorloc

Salinan Kaggle: https://www.kaggle.com/datasets/giantuji/UjiIndoorLoc

Lisensi dataset: Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/. Arsip data disertakan tanpa perubahan. Tabel penampang dan peta merupakan olahan dari dataset tersebut. Tidak ada dukungan atau pengesahan dari pembuat dataset yang diklaim.
