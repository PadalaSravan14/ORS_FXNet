import os
from dataclasses import dataclass, field
from typing import List


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")

FX_RATES_FILE = os.path.join(DATA_RAW_DIR, "Foreign_Exchange_Rates.csv")
GOLD_MACRO_FILE = os.path.join(DATA_RAW_DIR, "GoldUP.csv")

for _d in [DATA_RAW_DIR, DATA_PROCESSED_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(_d, exist_ok=True)

CURRENCY_PAIRS = ["USD", "EUR", "GBP", "JPY", "AUD"]
BASE_PAIR = "USD"  # USD/INR is the primary forecasting target

# Source column names inside Foreign_Exchange_Rates.csv (USD-denominated rates)
FX_SOURCE_COLUMNS = {
    "AUD": "AUSTRALIA - AUSTRALIAN DOLLAR/US$",
    "EUR": "EURO AREA - EURO/US$",
    "GBP": "UNITED KINGDOM - UNITED KINGDOM POUND/US$",
    "JPY": "JAPAN - YEN/US$",
}
DATE_COLUMN = "Time Serie"

TRAIN_START, TRAIN_END = "2000-01-01", "2012-12-31"
VAL_START, VAL_END = "2013-01-01", "2016-12-31"
TEST_START, TEST_END = "2017-01-01", "2019-12-31"

SLIDING_WINDOW = 60          # W, supervised input window (days)
SHORT_TERM_WINDOW = 21       # w_s
LONG_TERM_WINDOW = 180       # w_l, aggregated historical buffer
FORECAST_HORIZONS = [1, 3, 5, 10]

RANDOM_SEED = 42


@dataclass
class ShortTermEncoderConfig:
    hidden_units: int = 128
    num_layers: int = 2
    window: int = SHORT_TERM_WINDOW
    dropout: float = 0.2
    learning_rate: float = 1e-3


@dataclass
class LongTermEncoderConfig:
    hidden_units: int = 256
    num_layers: int = 3
    window: int = LONG_TERM_WINDOW
    dropout: float = 0.3
    layer_norm: bool = True


@dataclass
class AttentionFusionConfig:
    num_heads: int = 4
    attention_dim: int = 128
    use_film: bool = True


@dataclass
class MacroAutoencoderConfig:
    input_dim: int = 3  # gold, crude oil, inflation
    encoder_dims: List[int] = field(default_factory=lambda: [64, 32, 16])
    latent_dim: int = 16
    sensitivity_weight: float = 0.01     # lambda in Eq. (23)
    orthogonality_weight: float = 0.001


@dataclass
class FusionHeadConfig:
    layer_dims: List[int] = field(default_factory=lambda: [256, 128, 1])


@dataclass
class DQNConfig:
    state_dim: int = 22
    action_delta: float = 0.05           # {-delta, 0, +delta}
    replay_buffer_size: int = 10_000
    epsilon_decay: float = 0.995
    hidden_units: int = 64
    gamma: float = 0.95
    batch_size: int = 64
    learning_rate: float = 1e-3


@dataclass
class DDPGConfig:
    state_dim: int = 22
    action_bound: float = 0.15
    actor_hidden: int = 64
    critic_hidden: int = 128
    tau: float = 0.005
    gamma: float = 0.95
    learning_rate: float = 1e-3


@dataclass
class PPOConfig:
    state_dim: int = 22
    clip_ratio: float = 0.2
    hidden_units: int = 64
    entropy_coef: float = 0.01
    gamma: float = 0.95
    learning_rate: float = 3e-4


@dataclass
class TrainingConfig:
    optimizer: str = "adam"
    batch_size: int = 64
    epochs: int = 100
    early_stopping_patience: int = 10
    base_learning_rate: float = 3e-4
    huber_delta: float = 1.0
    rl_episodes: int = 500
    rl_episode_length: int = 20


DEFAULT_SHORT_TERM = ShortTermEncoderConfig()
DEFAULT_LONG_TERM = LongTermEncoderConfig()
DEFAULT_ATTENTION = AttentionFusionConfig()
DEFAULT_MACRO_AE = MacroAutoencoderConfig()
DEFAULT_FUSION_HEAD = FusionHeadConfig()
DEFAULT_DQN = DQNConfig()
DEFAULT_DDPG = DDPGConfig()
DEFAULT_PPO = PPOConfig()
DEFAULT_TRAINING = TrainingConfig()

TECHNICAL_INDICATOR_DIM = 6  # moving averages, RoC, rolling volatility, etc.
