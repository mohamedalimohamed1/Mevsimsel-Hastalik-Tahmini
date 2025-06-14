# Mevsimsel Hastalık Tahmini

Bu proje, mevsimsel hastalık verileri kullanılarak gelecekteki vaka sayılarını tahmin etmek amacıyla dört farklı derin öğrenme mimarisini (Vanilla LSTM, Stacked LSTM, Bidirectional LSTM, GRU) karşılaştırmalı olarak uygulayan bir zaman serisi modelleme çalışmasıdır.

##  Amaç

Mevsimsel hastalıkların gelecekteki yayılımını öngörerek halk sağlığı planlamalarına katkı sunmak ve farklı derin öğrenme yaklaşımlarının başarımını ölçmek.

##  Kullanılan Modeller

- Vanilla LSTM
- Stacked LSTM
- Bidirectional LSTM
- GRU

##  Proje Yapısı

.Mevsimsel Hastalık Tahmini
├── dataset/ (CSV olarak)
│ ├── dataset.cv
├── logs/ (CSV olarak)
│ ├── bidirectional_lstm/
│ ├── gru/
│ ├── stacked_lstm/
│ └── vanilla_lstm/
├── trained_models/ # Eğitilmiş modeller ve scaler objeleri
│ ├── bidirectional_lstm/
│ ├── gru/
│ ├── stacked_lstm/
│ └── vanilla_lstm/
├── visualization/ # Grafik çıktıları ve değerlendirme görselleri
│ ├── data_analysis/
│ ├── bidirectional_lstm/
│ ├── gru/
│ ├── stacked_lstm/
│ └── vanilla_lstm/
├── app.py # Ana başlatıcı dosya
├── data_analyze.py # Veri görselleştirme ve istatistiksel analiz
├── data_checking.py # Veri seti geçerlilik kontrolü
├── vanilla_lstm_forecasting.py
├── stacked_lstm_forecasting.py
├── bidirectional_lstm_forecasting.py
└── gru_forecasting.py


##  Kullanılan Değerlendirme Metrikleri

- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Square Error)
- **R² Score**
- **Accuracy Rate (%)**

##  Başlatmak İçin

1. Gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
dataset/dataset.csv dosyasını yerleştirin.

## İlgili tahmin scriptini çalıştırın:

python vanilla_lstm_forecasting.py
Diğer modeller için sırasıyla stacked_lstm_forecasting.py, bidirectional_lstm_forecasting.py ve gru_forecasting.py dosyalarını çalıştırabilirsiniz.

📈 Performans Karşılaştırma Tablosu
Model	MAE	RMSE	R² Score	Accuracy Rate (%)
Vanilla LSTM	147.91	185.71	0.0013	96.90
Stacked LSTM	96.63	125.38	0.0046	97.55
Bidirectional LSTM	88.74	118.10	0.0061	98.30
GRU	90.19	130.63	0.0047	96.87

Not: R² skorunun düşük olması sağlık verilerinin düzensiz yapısı ve ani mevsimsel değişimlerle açıklanabilir.

📅 Gelecek Sezon Tahmini – İlk 10 Hastalık
Vanilla LSTM
Disease	Predicted Count
Allergy	5383.64
Fever	4120.86
Cold	3979.54
Skin Rash	3933.78
Cough	3887.02
Dengue	3823.20
Headache	3693.97
Eye Infection	3671.31
Malaria	2756.76
Diarrhea	2362.37

Stacked LSTM
Disease	Predicted Count
Allergy	5285.14
Fever	4032.68
Cold	3920.89
Skin Rash	3885.90
Cough	3841.93
Dengue	3747.99
Headache	3668.47
Eye Infection	3639.96
Malaria	2763.91
Diarrhea	2043.63

Bidirectional LSTM
Disease	Predicted Count
Allergy	5164.83
Fever	4060.36
Cold	3921.59
Skin Rash	3871.37
Cough	3826.18
Dengue	3737.00
Headache	3696.65
Eye Infection	3690.99
Malaria	2582.33
Diarrhea	2316.93

GRU
Disease	Predicted Count
Allergy	5355.22
Fever	4095.92
Cold	3937.80
Skin Rash	3917.56
Cough	3884.03
Dengue	3746.56
Headache	3664.74
Eye Infection	3641.94
Malaria	2633.17
Diarrhea	2251.45

🔍 Gözlemler
Allergy hastalığı her modelde en fazla tahmin edilen vaka türü olmuştur.

Stacked LSTM ve Bidirectional LSTM, temel LSTM modeline göre daha düşük hata değerleri üretmiştir.

GRU, hata oranı düşük modellerden biri olup bazı sınıflarda sapmalar gözlenmiştir.

Genel doğruluk oranı tüm modellerde %96–98 aralığında seyretmiştir.
