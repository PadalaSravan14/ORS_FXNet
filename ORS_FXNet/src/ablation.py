
import copy

from config import (
    DEFAULT_SHORT_TERM, DEFAULT_LONG_TERM, DEFAULT_ATTENTION, DEFAULT_MACRO_AE,
    DEFAULT_FUSION_HEAD, DEFAULT_TRAINING,
)
from src.model import CurrencyForecastingNetwork
from src.train import train_neural_forecaster, generate_base_predictions
from src.evaluation_metrics import compute_all_metrics

ABLATION_VARIANTS = {
    "Full ORS-FXNet (Proposed)": {},
    "Without Reinforcement Learning Correction": {"skip_rl": True},
    "Without Macro Autoencoder (No Macro-Sensitivity)": {"no_macro_ae": True},
    "Without Attention Fusion (No Multi-head Attention)": {"no_attention": True},
    "Without Long-Term LSTM (Only Short-Term Encoder)": {"only_short_term": True},
    "Without Short-Term LSTM (Only Long-Term Encoder)": {"only_long_term": True},
    "Without Temporal Fusion (Independent Encoders Only)": {"no_temporal_fusion": True},
    "No Special Modules (Plain Single-Layer LSTM)": {"plain_lstm": True},
}


def run_ablation_study(n_features, n_macro_features, n_tech_features, pair,
                        train_loader, val_loader, test_loader,
                        short_window, long_window, macro_idx, tech_idx,
                        device="cpu", quick_epochs=None):
    results = {}
    trained_models = {}
    training_cfg = copy.deepcopy(DEFAULT_TRAINING)
    if quick_epochs is not None:
        training_cfg.epochs = quick_epochs

    for variant_name, ablation_flags in ABLATION_VARIANTS.items():
        model = CurrencyForecastingNetwork(
            n_features=n_features, n_macro_features=n_macro_features, n_tech_features=n_tech_features,
            currency_pairs=[pair],
            short_cfg=DEFAULT_SHORT_TERM, long_cfg=DEFAULT_LONG_TERM, attn_cfg=DEFAULT_ATTENTION,
            macro_cfg=DEFAULT_MACRO_AE, fusion_cfg=DEFAULT_FUSION_HEAD, ablation=ablation_flags,
        )
        model, _ = train_neural_forecaster(
            model, train_loader, val_loader, pair, training_cfg,
            short_window, long_window, macro_idx, tech_idx, device=device, verbose=False,
        )
        y_pred, y_true, _, _ = generate_base_predictions(
            model, test_loader, pair, short_window, long_window, macro_idx, tech_idx, device=device,
        )
        metrics = compute_all_metrics(y_true, y_pred)
        results[variant_name] = metrics
        trained_models[variant_name] = model

    return results, trained_models
