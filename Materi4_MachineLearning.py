
# Spam Email Detection dengan Naive Bayes

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# 1. Dataset simulasi
emails = [
    "Selamat anda memenangkan hadiah uang tunai",
    "Klik link ini untuk mendapatkan hadiah gratis",
    "Dapatkan bonus dan uang gratis sekarang",
    "Penawaran khusus beli sekarang",
    "Anda terpilih menerima hadiah besar",
    "Klaim hadiah anda hari ini",
    "Rapat tim dimulai pukul sepuluh pagi",
    "Berikut laporan praktikum yang diminta",
    "Jadwal kuliah minggu depan sudah dibagikan",
    "Tolong kirim tugas melalui email",
    "Besok kita mengadakan rapat proyek",
    "Dokumen laporan sudah saya lampirkan",
    "Dapatkan diskon besar untuk produk ini",
    "Anda memenangkan voucher belanja gratis",
    "Mohon periksa dokumen sebelum rapat",
    "Pengumpulan tugas ditutup hari Jumat",
    "Promo terbatas segera pesan sekarang",
    "Catatan rapat sudah dikirim ke tim",
    "Klaim bonus gratis dengan klik tautan",
    "Jangan lupa hadir dalam perkuliahan"
]

labels = [
    "spam", "spam", "spam", "spam", "spam", "spam",
    "ham", "ham", "ham", "ham", "ham", "ham",
    "spam", "spam", "ham", "ham", "spam", "ham",
    "spam", "ham"
]

# 2. Bagi data untuk pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(
    emails,
    labels,
    test_size=0.30,
    random_state=42,
    stratify=labels
)

# 3. Buat pipeline TF-IDF dan Naive Bayes
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

# 4. Latih model
model.fit(X_train, y_train)

# 5. Evaluasi model
predictions = model.predict(X_test)

print("Akurasi:", accuracy_score(y_test, predictions))
print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

# 6. Uji email baru
email_baru = [
    "Selamat anda mendapatkan hadiah gratis",
    "Mohon kirim laporan rapat hari ini"
]

hasil = model.predict(email_baru)

for email, label in zip(email_baru, hasil):
    print(f"\nEmail: {email}")
    print(f"Prediksi: {label}")
