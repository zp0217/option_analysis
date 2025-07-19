#code run via colab,local machine takes very longtime**

#import packages
#packages
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,SimpleRNN, Dense,Input,LSTM,GRU
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import GroupKFold
import seaborn as sns
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import plotly.graph_objects as go

#data cleaning:** 

options_path = "ndx_option_raw.csv"
options = pd.read_csv(options_path)


options['date']  = pd.to_datetime(options['date'])
options['exdate'] = pd.to_datetime(options['exdate'])
options['mid_price'] = (options['best_bid'] + options['best_offer']) / 2
#expire date
options['days_to_expiry'] = (options['exdate'] - options['date']).dt.days
# 5.3: encode call/put flag
options['cp_flag_encoded'] = options['cp_flag'].map({'C': 1, 'P': 0})

# divide strike price by 1000( strike prices recorded  strike price * 1000)
options['strike_price'] = (options['strike_price'])/1000

# Step 8: Drop rows with missing critical values
required_cols = [
    'strike_price','cp_flag_encoded',
   'days_to_expiry','impl_volatility','delta','gamma','vega','theta','mid_price'
]
df = options.dropna(subset=required_cols)

# Step 9: Select features and target
features = [
    'strike_price','cp_flag_encoded',
   'days_to_expiry','mid_price','delta','gamma','vega','theta'
]
X = df[features]
y= df['impl_volatility']
groups = df['date'].values  # group by trading date
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



#RNN(group kfold = 3)

X = df[features].values
X = df[features].values
y = df["impl_volatility"].values
groups = df["date"].values
X_rnn = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

gkf = GroupKFold(n_splits=3 )
mse_scores = []
r2_scores  = []
histories  = []
final_pred = final_true = None

for fold, (train_idx, test_idx) in enumerate(gkf.split(X_rnn, y, groups), start=1):
    print(f"\n--- RNN Fold {fold} ---")
    X_tr, X_te = X_rnn[train_idx], X_rnn[test_idx]
    y_tr, y_te = y[train_idx],   y[test_idx]


    model = Sequential([
        Input(shape=(1, X_rnn.shape[2])),
        SimpleRNN(
            1024,
            activation="tanh",
            kernel_regularizer=l2(1e-4),
            recurrent_regularizer=l2(1e-4),
            return_sequences=False
        ),
        Dropout(0.1),
        Dense(512, activation="relu", kernel_regularizer=l2(1e-4)),
        Dropout(0.1),
        Dense(256, activation="relu", kernel_regularizer=l2(1e-4)),
        Dropout(0.1),
        Dense(1, activation="linear")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse"
    )


    early_stop = EarlyStopping(
        monitor="val_loss", patience=5,
        restore_best_weights=True, verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5,
        patience=5, min_lr=1e-5, verbose=1
    )

    history = model.fit(
        X_tr, y_tr,
        epochs= 30,
        batch_size=1024,
        validation_split=0.1,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    histories.append(history)

    y_pred = model.predict(X_te, verbose=0).flatten()
    mse = mean_squared_error(y_te, y_pred)
    r2  = r2_score(y_te, y_pred)
    print(f"Fold {fold} — MSE: {mse:.5f}, R²: {r2:.4f}")

    mse_scores.append(mse)
    r2_scores.append(r2)

    if fold == 3:
        final_pred, final_true = y_pred, y_te


print("\nRNN Cross‑Validation Results:")
print("MSE scores:", np.round(mse_scores,5))
print("R² scores :", np.round(r2_scores,4))
print(f"Average MSE: {np.mean(mse_scores):.5f}")
print(f"Average R² : {np.mean(r2_scores):.4f}")

if fold == 3:
    y_test_rnn = y_te
    y_pred_rnn = y_pred
    mse_rnn = mse_scores
    r2_rnn  = r2_scores



#RNN visualization:

#Training vs Validation Loss (final fold)
hist = histories[-1]
plt.figure(figsize=(8,4))
plt.plot(hist.history["loss"], label="Train Loss")
plt.plot(hist.history["val_loss"], label="Val Loss")
plt.title("Final Fold: RNN Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#Predicted vs Actual
plt.figure(figsize=(6,6))
plt.scatter(final_true, final_pred, alpha=0.3, edgecolor="k", linewidth=0.5)
plt.plot(
    [final_true.min(), final_true.max()],
    [final_true.min(), final_true.max()],
    "r--", label="Ideal"
)
plt.title("Final Fold: RNN Predicted vs Actual IV")
plt.xlabel("Actual IV")
plt.ylabel("Predicted IV")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#LSTM(group kfold = 3)
X = df[features].values
y = df["impl_volatility"].values
groups = df["date"].values 

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))


gkf = GroupKFold(n_splits=3)
mse_scores = []
r2_scores  = []

for fold, (train_idx, test_idx) in enumerate(gkf.split(X_lstm, y, groups), start=1):
    print(f"\n--- LSTM Fold {fold} ---")
    X_train, X_test = X_lstm[train_idx], X_lstm[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]


    model = Sequential([
        Input(shape=(1, X_lstm.shape[2])),
        LSTM(
            1024,
            activation="tanh",
            kernel_regularizer=l2(1e-4),
            recurrent_regularizer=l2(1e-4),
            return_sequences=False
        ),
        Dropout(0.1),
        Dense(512, activation="relu", kernel_regularizer=l2(1e-4)),
        Dropout(0.1),
        Dense(256, activation="relu", kernel_regularizer=l2(1e-4)),
        Dropout(0.1),
        Dense(1, activation="linear")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse"
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-5,
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=1024,
        validation_split=0.1,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    y_pred = model.predict(X_test, verbose=0).flatten()
    mse = mean_squared_error(y_test, y_pred)
    r2  = r2_score(y_test,  y_pred)
    mse_scores.append(mse)
    r2_scores.append(r2)
    print(f"Fold {fold} — Test MSE: {mse:.5f}, Test R²: {r2:.4f}")


print("\nLSTM Cross-Validation Results (3 folds):")
print("MSE scores:", np.round(mse_scores, 5))
print("R²  scores:", np.round(r2_scores, 4))
print(f"Average MSE: {np.mean(mse_scores):.5f}")
print(f"Average R² : {np.mean(r2_scores):.4f}")

if fold == 3:
    y_test_lstm = y_test
    y_pred_lstm = y_pred
    mse_lstm = mse_scores
    r2_lstm  = r2_scores

#LSTM visualization

# Train vs Validation Loss (final fold)
plt.figure(figsize=(8, 4))
plt.plot(history.history["loss"], label="Train Loss", linewidth=2)
plt.plot(history.history["val_loss"], label="Validation Loss", linewidth=2)
plt.title("Final Fold: LSTM Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Predicted vs Actual Plot (final fold)
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.3, edgecolor='k', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Ideal")
plt.title("Final Fold: LSTM Predicted vs Actual IV")
plt.xlabel("Actual Implied Volatility")
plt.ylabel("Predicted Implied Volatility")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#GRU(group kfold = 3)
X = df[features].values
y = df["impl_volatility"].values
groups = df["date"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_gru = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))


gkf = GroupKFold(n_splits=3)
gru_mse, gru_r2 = [], []
histories = []
final_pred = final_true = None

for fold, (trn_idx, val_idx) in enumerate(gkf.split(X_gru, y, groups), start=1):
    print(f"\n--- GRU Fold {fold} ---")
    X_tr, X_val = X_gru[trn_idx], X_gru[val_idx]
    y_tr, y_val = y[trn_idx],   y[val_idx]

    model = Sequential([
        Input(shape=(1, X_gru.shape[2])),
        GRU(1024, activation="tanh",
            kernel_regularizer=l2(1e-4),
            recurrent_regularizer=l2(1e-4)),
        Dropout(0.1),
        Dense(512, activation="relu", kernel_regularizer=l2(1e-4)),
        Dropout(0.1),
        Dense(256, activation="relu", kernel_regularizer=l2(1e-4)),
        Dropout(0.1),
        Dense(1, activation="linear")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse"
    )

    early_stop = EarlyStopping(
        monitor="val_loss", patience=6,
        restore_best_weights=True, verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5,
        patience=3, min_lr=1e-5, verbose=1
    )


    history = model.fit(
        X_tr, y_tr,
        epochs=30,
        batch_size=1024,
        validation_split=0.1,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )


    y_pred = model.predict(X_val, verbose=0).flatten()
    mse = mean_squared_error(y_val, y_pred)
    r2  = r2_score(y_val, y_pred)
    print(f"Fold {fold} — MSE: {mse:.5f}, R²: {r2:.4f}")

    gru_mse.append(mse)
    gru_r2.append(r2)

   
    if fold == 3:
        histories.append(history)
        final_pred, final_true = y_pred, y_val

print("\nGRU Cross‑Validation Results:")
print("MSE scores:", np.round(gru_mse,5))
print("R² scores :", np.round(gru_r2,4))
print(f"Avg MSE: {np.mean(gru_mse):.5f}")
print(f"Avg R² : {np.mean(gru_r2):.4f}")


if fold == 3:
    y_test_gru = final_true
    y_pred_gru = final_pred
    mse_gru = gru_mse
    r2_gru  = gru_r2

#GRU visualization

#Train vs Val Loss (final fold)
hist = histories[0]
plt.figure(figsize=(8,4))
plt.plot(hist.history["loss"], label="Train Loss")
plt.plot(hist.history["val_loss"], label="Val Loss")
plt.title("Final Fold: GRU Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Predicted vs Actual
plt.figure(figsize=(6,6))
plt.scatter(final_true, final_pred, alpha=0.3, edgecolor="k", linewidth=0.5)
plt.plot(
    [final_true.min(), final_true.max()],
    [final_true.min(), final_true.max()],
    "r--", label="Ideal"
)
plt.title("Final Fold: GRU Predicted vs Actual IV")
plt.xlabel("Actual IV")
plt.ylabel("Predicted IV")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#comparing all models 
models = ['RNN', 'LSTM', 'GRU']
avg_mse = [np.mean(mse_rnn), np.mean(mse_lstm), np.mean(mse_gru)]
avg_r2  = [np.mean(r2_rnn),  np.mean(r2_lstm),  np.mean(r2_gru)]

x = np.arange(len(models))
width = 0.35

fig, ax1 = plt.subplots(figsize=(8,4))
ax2 = ax1.twinx()

bars1 = ax1.bar(x - width/2, avg_mse, width, label='Avg MSE', color='skyblue')
bars2 = ax2.bar(x + width/2, avg_r2, width, label='Avg R²', color='salmon')

ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.set_ylabel('Average MSE')
ax2.set_ylabel('Average R²')
ax1.set_title('Model Performance Comparison (3‑Fold)')

ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.tight_layout()
plt.show()



fig, axes = plt.subplots(1, 3, figsize=(15,4), sharex=True, sharey=True)

for ax, (name, y_true, y_pred) in zip(axes, [
    ('RNN',   y_test_rnn,   y_pred_rnn),
    ('LSTM',  y_test_lstm,  y_pred_lstm),
    ('GRU',   y_test_gru,   y_pred_gru),
]):
    ax.scatter(y_true, y_pred, alpha=0.3, edgecolor='w', s=20)
    mn, mx = np.min([y_true.min(), y_pred.min()]), np.max([y_true.max(), y_pred.max()])
    ax.plot([mn,mx],[mn,mx],'r--', lw=1)
    ax.set_title(f'{name}: Pred vs Actual')
    ax.set_xlabel('Actual IV')
    ax.set_ylabel('Predicted IV' if ax is axes[0] else '')
    ax.grid(alpha=0.3)

fig.suptitle('Final Fold: Predicted vs Actual Implied Volatility', y=1.02)
plt.tight_layout()
plt.show()

#preparing for time series plot for result from GRU model

for fold, (trn_idx, val_idx) in enumerate(gkf.split(X_gru, y, groups), start=1):
    if fold == 3:
        final_dates = df.iloc[val_idx]["date"].values
        final_cp_flag = df.iloc[val_idx]["cp_flag"].values
        final_strike_price = df.iloc[val_idx]["strike_price"].values
        break

df_test_gru = pd.DataFrame({
    "date": pd.to_datetime(final_dates),
    "impl_volatility": y_test_gru,
    "predicted_iv": y_pred_gru,
    "cp_flag": final_cp_flag,
    "strike_price": final_strike_price
})


#TIME SERIES PLOT FOR PUT OPTION
import pandas as pd
import plotly.graph_objects as go
df = pd.read_csv('data/processed-data/gru.csv')
df["date"] = pd.to_datetime(df["date"])
df = df[df["cp_flag"] == "P"]

selected_strikes = sorted(df["strike_price"].unique())[:5]
df = df[df["strike_price"].isin(selected_strikes)]

fig = go.Figure()

for strike in selected_strikes:
    subset = df[df["strike_price"] == strike].sort_values("date")

    fig.add_trace(go.Scatter(
        x=subset["date"], y=subset["impl_volatility"],
        mode="lines+markers",
        name=f"Actual IV - {int(strike)}",
        line=dict(dash="solid")
    ))

    fig.add_trace(go.Scatter(
        x=subset["date"], y=subset["predicted_iv"],
        mode="lines+markers",
        name=f"Predicted IV - {int(strike)}",
        line=dict(dash="dash")
    ))

fig.update_layout(
    title="Actual vs Predicted Implied Volatility (Put Option)",
    xaxis_title="Date",
    yaxis_title="Implied Volatility",
    hovermode="x unified",
    height=600
)

fig.show()

#TIME SERIES PLOT FOR CALL OPTION
import pandas as pd
import plotly.graph_objects as go
df = pd.read_csv('data/processed-data/gru.csv')
df["date"] = pd.to_datetime(df["date"])
df = df[df["cp_flag"] == "C"]

selected_strikes = sorted(df["strike_price"].unique())[:5]
df = df[df["strike_price"].isin(selected_strikes)]

fig = go.Figure()

for strike in selected_strikes:
    subset = df[df["strike_price"] == strike].sort_values("date")

    fig.add_trace(go.Scatter(
        x=subset["date"], y=subset["impl_volatility"],
        mode="lines+markers",
        name=f"Actual IV - {int(strike)}",
        line=dict(dash="solid")
    ))

    fig.add_trace(go.Scatter(
        x=subset["date"], y=subset["predicted_iv"],
        mode="lines+markers",
        name=f"Predicted IV - {int(strike)}",
        line=dict(dash="dash")
    ))

fig.update_layout(
    title="Actual vs Predicted Implied Volatility (Call Option)",
    xaxis_title="Date",
    yaxis_title="Implied Volatility",
    hovermode="x unified",
    height=600
)

fig.show()
