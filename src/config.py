"""
Konfigurasi global untuk aplikasi SIKAP.
Semua konstanta dan mapping terpusat di sini.
"""

# Model & API
GEMINI_MODEL = "gemini-2.0-flash"  # Flash 2.5 tersedia, pakai flash untuk kecepatan
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # Dimensi output MiniLM

# Classification settings
TOP_K_FAISS = 30       # Jumlah kandidat dari FAISS per level
TOP_K_OUTPUT = 3        # Jumlah rekomendasi akhir
SIMILARITY_THRESHOLD = 0.35  # Threshold minimum similarity

# Kode primer (level 1) — digunakan untuk referensi
PRIMER_CODES = ["000", "100", "200", "300", "400", "500", "600", "700", "800", "900"]

# Mapping kode primer → nama
PRIMER_NAMES = {
    "000": "Umum",
    "100": "Pemerintahan",
    "200": "Politik",
    "300": "Keamanan dan Ketertiban",
    "400": "Kesejahteraan Rakyat",
    "500": "Perekonomian",
    "600": "Pekerjaan Umum dan Ketenagakerjaan",
    "700": "Pengawasan",
    "800": "Kepegawaian",
    "900": "Keuangan",
}
