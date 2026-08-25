
import numpy as np
import torch
import torch.nn as nn
import warnings

warnings.filterwarnings("ignore")

def fit_arima_forecast(train_series, test_length, order=(2, 1, 2)):
    from statsmodels.tsa.arima.model import ARIMA
    history = list(train_series)
    predictions = []
    model_fit = ARIMA(history, order=order).fit()
    for step in range(test_length):
        forecast = model_fit.forecast(steps=1)[0]
        predictions.append(forecast)
        # roll forward: refit lazily every 20 steps for tractability
        if (step + 1) % 20 == 0:
            history.append(forecast)
            try:
                model_fit = ARIMA(history, order=order).fit()
            except Exception:
                pass
    return np.array(predictions[:test_length])


def fit_var_forecast(train_df, test_length, target_col, maxlags=5):
    from statsmodels.tsa.api import VAR
    model = VAR(train_df.values)
    fitted = model.fit(maxlags=maxlags)
    target_idx = list(train_df.columns).index(target_col)
    lag_order = fitted.k_ar
    history = train_df.values[-lag_order:]
    predictions = []
    for _ in range(test_length):
        forecast = fitted.forecast(history, steps=1)
        predictions.append(forecast[0, target_idx])
        history = np.vstack([history[1:], forecast])
    return np.array(predictions)


class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                             dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class BiLSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                             dropout=dropout if num_layers > 1 else 0.0, bidirectional=True)
        self.head = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class GRUForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class ConvolutionalLSTMForecaster(nn.Module):
    """1D convolutional feature extractor feeding an LSTM temporal head."""

    def __init__(self, input_dim, conv_channels=64, hidden_dim=128, num_layers=1, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, conv_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.activation = nn.ReLU()
        self.lstm = nn.LSTM(conv_channels, hidden_dim, num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (B, T, F) -> conv expects (B, F, T)
        h = self.activation(self.conv(x.transpose(1, 2))).transpose(1, 2)
        out, _ = self.lstm(h)
        return self.head(out[:, -1, :]).squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerForecaster(nn.Module):
    def __init__(self, input_dim, model_dim=128, num_heads=4, num_layers=2, ff_dim=256, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.pos_encoding = PositionalEncoding(model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(model_dim, 1)

    def forward(self, x):
        h = self.pos_encoding(self.input_proj(x))
        h = self.encoder(h)
        return self.head(h[:, -1, :]).squeeze(-1)


BASELINE_REGISTRY = {
    "LSTM": LSTMForecaster,
    "BiLSTM": BiLSTMForecaster,
    "GRU": GRUForecaster,
    "CNN-LSTM": ConvolutionalLSTMForecaster,
    "Transformer": TransformerForecaster,
}
