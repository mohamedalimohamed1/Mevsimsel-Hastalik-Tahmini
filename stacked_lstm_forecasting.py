import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, Callback
import joblib

# ----------------------------
# Create Required Folders
# ----------------------------
base_dirs = ["visualization/stacked_lstm", "logs/stacked_lstm", "trained_models/stacked_lstm"]
for dir_path in base_dirs:
    os.makedirs(dir_path, exist_ok=True)

# ----------------------------
# 1. Load & Prepare Dataset
# ----------------------------
df = pd.read_csv("./dataset/dataset.csv")
df['SeasonOrder'] = df['Season'].map({'Spring': 0, 'Summer': 1, 'Autumn': 2, 'Winter': 3})
le_disease = LabelEncoder()
df['Disease_enc'] = le_disease.fit_transform(df['Disease'])

grouped = df.groupby(['Disease_enc', 'Year', 'SeasonOrder'])['Disease_Count']\
            .sum().reset_index().sort_values(['Disease_enc','Year','SeasonOrder'])

seq_len, pred_steps = 4, 1
X_all, y_all, labels = [], [], []
for d in grouped['Disease_enc'].unique():
    vals = grouped[grouped['Disease_enc'] == d]['Disease_Count'].values
    if len(vals) >= seq_len + pred_steps:
        for i in range(len(vals) - seq_len - pred_steps + 1):
            X_all.append(vals[i:i+seq_len])
            y_all.append(vals[i+seq_len])
            labels.append(d)

X_all, y_all, labels = map(np.array, (X_all, y_all, labels))

# ----------------------------
# 2. Scaling
# ----------------------------
sc_X = MinMaxScaler()
sc_y = MinMaxScaler()
X_scaled = sc_X.fit_transform(X_all)
y_scaled = sc_y.fit_transform(y_all.reshape(-1, 1)).flatten()

# ----------------------------
# 3. Train/Test Split
# ----------------------------
split = int(len(X_scaled) * 0.8)
X_train, X_test = X_scaled[:split], X_scaled[split:]
y_train, y_test = y_scaled[:split], y_scaled[split:]
labels_test = labels[split:]

X_train = X_train.reshape((-1, seq_len, 1))
X_test = X_test.reshape((-1, seq_len, 1))

# ----------------------------
# 4. Stacked LSTM Model
# ----------------------------
class CustomLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch+1:3d} | loss={logs['loss']:.4f} | val_loss={logs['val_loss']:.4f}")

model = Sequential([
    LSTM(64, activation='relu', return_sequences=True, input_shape=(seq_len, 1)),
    LSTM(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

history = model.fit(
    X_train, y_train,
    epochs=100,
    validation_data=(X_test, y_test),
    callbacks=[
        CustomLogger(),
        CSVLogger("logs/stacked_lstm/train_log.csv", append=False),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ],
    verbose=0
)

# ----------------------------
# 5. Save Model & Scalers
# ----------------------------
model.save("trained_models/stacked_lstm/model.h5")
joblib.dump(sc_X, "trained_models/stacked_lstm/scaler_X.pkl")
joblib.dump(sc_y, "trained_models/stacked_lstm/scaler_y.pkl")
joblib.dump(le_disease, "trained_models/stacked_lstm/label_encoder.pkl")

# ----------------------------
# 6. Evaluation
# ----------------------------
y_pred_scaled = model.predict(X_test).flatten()
y_pred = sc_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_true = sc_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)
accuracy = 100 - (mae / (np.mean(y_true) + 1e-6)) * 100

print("\n=== Evaluation (Stacked LSTM) ===")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")
print(f"Accuracy Rate: {accuracy:.2f}%")

# Save evaluation metrics
metrics_df = pd.DataFrame({
    'Metric': ['MAE', 'RMSE', 'R2 Score', 'Accuracy Rate (%)'],
    'Value': [mae, rmse, r2, accuracy]
})
metrics_df.to_csv("visualization/stacked_lstm/evaluation_metrics.csv", index=False)

# Plot metrics
plt.figure(figsize=(8, 5))
plt.bar(metrics_df['Metric'], metrics_df['Value'], color=['dimgray', 'slategray', 'black', 'gray'])
plt.title("Stacked LSTM Evaluation Metrics")
plt.ylabel("Score")
plt.grid(True, linestyle='--', color='darkgray')
plt.tight_layout()
plt.savefig("visualization/stacked_lstm/metrics_plot.png")
plt.close()

# ----------------------------
# 7. Predict Next Season
# ----------------------------
latest = grouped.groupby('Disease_enc').tail(seq_len)
results = []
for d in latest['Disease_enc'].unique():
    vals = latest[latest['Disease_enc'] == d]['Disease_Count'].values
    if len(vals) == seq_len:
        xs = sc_X.transform(vals.reshape(1, -1)).reshape((1, seq_len, 1))
        pred = model.predict(xs)[0][0]
        count = sc_y.inverse_transform([[pred]])[0][0]
        results.append((le_disease.inverse_transform([d])[0], count))

df_out = pd.DataFrame(results, columns=["Disease", "Predicted_Count"]).sort_values("Predicted_Count", ascending=False)
df_out.to_csv("visualization/stacked_lstm/forecasted_diseases_next_season.csv", index=False)

# Plot prediction comparison
plt.figure(figsize=(12, 6))
plt.bar(df_out['Disease'], df_out['Predicted_Count'], color='darkslategray')
plt.title("Stacked LSTM - Forecasted Disease Counts for Next Season")
plt.ylabel("Predicted Case Count")
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', color='gray')
plt.tight_layout()
plt.savefig("visualization/stacked_lstm/forecasted_counts_plot.png")
plt.close()

print("\n=== Forecasted Disease Counts (Next Season - Stacked LSTM) ===")
print(df_out.head(10))
