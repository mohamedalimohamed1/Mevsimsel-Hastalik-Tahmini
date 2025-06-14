import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Stil ayarları
sns.set(style="whitegrid")

# Klasör oluştur
output_dir = "visualization/data_analysis"
os.makedirs(output_dir, exist_ok=True)

# Veriyi yükle
df = pd.read_csv("./dataset/dataset.csv")

# Tarih sütununu datetime formatına çevir
df["Date"] = pd.to_datetime(df["Date"])

# Yıl, Ay ve Mevsim bilgileri çıkar
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month

# Mevsim hesaplama fonksiyonu
def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"

df["Season"] = df["Month"].apply(get_season)

# ---------- 1. Toplam Vaka Sayısına Göre Hastalıklar (Line Plot) ----------
disease_sum = df.groupby("Disease")["Disease_Count"].sum().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.lineplot(x=disease_sum.index, y=disease_sum.values, marker="o", linewidth=2.5, color="teal")
plt.title("Toplam Vaka Sayısına Göre Hastalıklar (Çizgi Grafiği)")
plt.xlabel("Hastalık Türü")
plt.ylabel("Toplam Vaka Sayısı")
plt.xticks(rotation=45)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "disease_distribution_lineplot.png"))
plt.close()

# ---------- 2. Yıllara Göre Hastalık Bazlı Vaka Sayısı (Line Plot) ----------
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=df,
    x="Year",
    y="Disease_Count",
    hue="Disease",
    estimator="sum",
    ci=None,
    marker="o"
)
plt.title("Yıllara Göre Hastalık Bazlı Vaka Sayısı")
plt.xlabel("Yıl")
plt.ylabel("Toplam Vaka Sayısı")
plt.legend(title="Hastalık", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "yearly_disease_trends.png"))
plt.close()

# ---------- 3. Mevsimlere Göre Vaka Dağılımı (Boxplot) ----------
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="Season", y="Disease_Count", palette="coolwarm")
plt.title("Mevsimlere Göre Vaka Sayısı Dağılımı")
plt.xlabel("Mevsim")
plt.ylabel("Vaka Sayısı")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "seasonal_distribution.png"))
plt.close()

print("Tüm grafikler başarıyla 'visualization/data_analysis' klasörüne kaydedildi.")
