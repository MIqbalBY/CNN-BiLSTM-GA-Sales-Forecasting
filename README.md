# CNN-BiLSTM-GA Sales Forecasting

> **Tugas Akhir – ES234733**
> Institut Teknologi Sepuluh Nopember · Departemen Sistem Informasi · FTEIC

Peramalan penjualan mingguan produk mie instan dari produsen ke distributor menggunakan model hibrida **CNN–BiLSTM** yang dioptimasi dengan **Genetic Algorithm (GA)**. Studi kasus: PT. XYZ Banjarmasin, data 2019–2025.

---

## Identitas

| | |
|---|---|
| **Judul** | Peramalan Penjualan Produk Mie Instan Menggunakan Model Hibrida CNN–BiLSTM dengan Optimasi Genetic Algorithm |
| **Penulis** | Muhammad Iqbal Baiduri Yamani |
| **NRP** | 5026221103 |
| **Dosen Pembimbing** | Edwin Riksakomara, S.Kom., M.T. |
| **Program Studi** | S-1 Sistem Informasi, FTEIC – ITS |
| **Tahun** | 2025 |

---

## Abstrak

Industri *Fast Moving Consumer Goods* (FMCG) menghadapi tantangan dalam memprediksi permintaan mingguan secara akurat akibat pola yang kompleks dan dinamis. Kesalahan peramalan berujung pada kelebihan stok (meningkatkan biaya penyimpanan) atau kekurangan stok (kehilangan penjualan). Teknik konvensional tidak memadai untuk menangkap pola nonlinier dalam data *time-series*.

Penelitian ini mengusulkan model hibrida **CNN-BiLSTM** yang disempurnakan melalui **Genetic Algorithm** untuk meramalkan penjualan mingguan FMCG dari produsen ke distributor. Pipeline meliputi:
- Pengumpulan data penjualan mingguan 2019–2025
- Pembersihan data dan penanganan outlier via IQR clipping
- Transformasi *time-series* ke format *supervised learning* menggunakan *sliding window* lag-8
- CNN untuk ekstraksi pola jangka pendek, BiLSTM untuk pola jangka panjang
- GA + K-Fold Cross Validation (k=5) untuk optimasi hiperparameter
- Evaluasi dengan RMSE, MAE, MAPE, dan R²

**Kata kunci:** CNN-BiLSTM · FMCG · Genetic Algorithm · Peramalan · Rantai Pasok · Time Series

---

## Arsitektur Model

```
Input: X ∈ ℝ⁸  (8-week sliding window)
         │
  ┌──────▼────────────────────────────────────────────┐
  │  Conv1D (ReLU)                                    │
  │  · filters & kernel_size dioptimasi GA            │  ← Deteksi pola lokal temporal
  └──────┬────────────────────────────────────────────┘
         │
  ┌──────▼────────────────────────────────────────────┐
  │  MaxPooling1D                                     │
  │  · pool_size dioptimasi GA                        │  ← Kompresi fitur dominan
  └──────┬────────────────────────────────────────────┘
         │
  ┌──────▼──────┐
  │   Flatten   │  ← Reshape 2D → 1D sebelum masuk BiLSTM
  └──────┬──────┘
         │
  ┌──────▼────────────────────────────────────────────┐
  │  BiLSTM                                           │
  │  · units dioptimasi GA                            │  ← Temporal modeling dua arah
  │  · forward  : minggu 1 → 8 (tren masa lalu)      │     (forward + backward)
  │  · backward : minggu 8 → 1 (sinyal akhir periode)│
  └──────┬────────────────────────────────────────────┘
         │
  ┌──────▼────────────────────────────────────────────┐
  │  Dropout                                          │
  │  · rate dioptimasi GA (0.2 – 0.5)                 │  ← Cegah overfitting
  └──────┬────────────────────────────────────────────┘
         │
  ┌──────▼──────────────────────────────────┐
  │  Dense (1 neuron, linear activation)    │  ← Output: ŷ ∈ ℝ¹ (prediksi minggu t+1)
  └─────────────────────────────────────────┘
```

**Hiperparameter yang dioptimasi Genetic Algorithm:**

| Parameter | Deskripsi |
|---|---|
| `filters` | Jumlah filter Conv1D |
| `kernel_size` | Ukuran kernel Conv1D |
| `pool_size` | Ukuran pool MaxPooling1D |
| `bilstm_units` | Jumlah unit BiLSTM |
| `dropout_rate` | Dropout rate (0.2–0.5) |
| `learning_rate` | Learning rate Adam optimizer |
| `batch_size` | Ukuran batch training |

**Loss function:** MSE · **Optimizer:** Adam · **Aktivasi hidden:** ReLU · **Aktivasi output:** Linear

---

## Pipeline Notebook

Pipeline terdiri dari 19 notebook terstruktur:

| No | Notebook | Deskripsi |
|---|---|---|
| 00 | Project Scope & Reproducibility | Setup seed, environment lock |
| 01 | Data Load & Schema Validation | Load data mentah, validasi kolom |
| 02 | Time Index Integrity Audit | Cek duplikasi & missing week |
| 03 | EDA Univariate Profile | Distribusi, trend, seasonality |
| 04 | Outlier Handling – IQR Clipping | Deteksi & clipping outlier |
| 05 | Sliding Window Supervised Framing | Buat lag features (τ = 8) |
| 06 | Split Builder (60:40 / 70:30 / 80:20 / 90:10) | Buat semua proporsi split |
| 07 | Walk-Forward CV Builder | Setup walk-forward cross validation |
| 08 | CNN Baseline – All Splits | Baseline CNN murni |
| 09 | CNN + GA – All Splits | CNN + optimasi GA |
| 10 | BiLSTM Baseline – All Splits | Baseline BiLSTM murni |
| 11 | BiLSTM + GA – All Splits | BiLSTM + optimasi GA |
| 12 | CNN-BiLSTM Baseline – All Splits | Hybrid CNN-BiLSTM murni |
| 13 | CNN-BiLSTM + GA – All Splits | **Model utama** – CNN-BiLSTM + GA |
| 14 | Traditional Baselines – All Splits | ARIMA, Holt-Winters, dll. |
| 15 | Comprehensive Comparison & Ranking | Komparasi seluruh model |

---

## Eksperimen & Evaluasi

**Model yang dibandingkan:**
- CNN · CNN+GA
- BiLSTM · BiLSTM+GA
- CNN-BiLSTM · **CNN-BiLSTM+GA** ← model utama

**Proporsi split data:**
- 60:40 · 70:30 · 80:20 · 90:10

**Metrik evaluasi:**

| Metrik | Keterangan |
|---|---|
| MAPE | Metrik utama pemilihan model terbaik |
| MAE | Mean Absolute Error |
| RMSE | Root Mean Square Error |
| R² | Koefisien determinasi |

---

## Struktur Repository

```
CNN-BiLSTM-GA-Sales-Forecasting/
├── data/               # Dataset penjualan mingguan PT. XYZ (2019–2025)
├── notebook/           # 19 notebook pipeline (00–18)
├── markdown/           # Dokumen tugas akhir & pipeline spec
├── .gitignore
└── README.md
```

---

## Lisensi

Lihat [LICENSE](LICENSE).
