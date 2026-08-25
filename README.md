# ORS-FXNet — How to Run

This repository contains the full training/evaluation pipeline. This guide
only covers installation and usage.

## 1. Requirements

- Python 3.10+
- pip

## 2. Setup

```bash
cd ORS_FXNet

python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## 3. Data

Two CSV files must be present in `data/raw/` (already included in this
package):

```
data/raw/Foreign_Exchange_Rates.csv
data/raw/GoldUP.csv
```

If either file is missing, running the pipeline will stop immediately with
a `FileNotFoundError` telling you which file to add. Nothing is generated
automatically — only these two files are used.

## 4. Running the pipeline

From the project root:

```bash
# Fast smoke test — few epochs / RL episodes, just to confirm everything runs
python run_pipeline.py --quick

# Full run — 100 training epochs, 500 RL episodes per currency pair
python run_pipeline.py

# Only train specific currency pairs (default is all five: USD EUR GBP JPY AUD)
python run_pipeline.py --pairs USD EUR JPY

# Custom epoch / RL-episode budget
python run_pipeline.py --epochs 50 --rl-episodes 200

# Force CPU or GPU explicitly
python run_pipeline.py --device cpu
python run_pipeline.py --device cuda
```

`--device` defaults to `cuda` automatically if a GPU is available, otherwise
`cpu`.

### What this does

1. Loads and preprocesses the two CSV files.
2. Generates all plots and result tables.
3. Trains the baseline models (ARIMA, VAR, LSTM, BiLSTM, GRU, CNN-LSTM,
   Transformer).
4. Trains ORS-FXNet (neural base + DQN/DDPG/PPO correction ensemble) for
   each selected currency pair.
5. Runs statistical significance tests, the ablation study, walk-forward
   cross-validation, and multi-horizon forecasting.
6. Saves every trained model, the fitted scalers, and all figures/tables.

A full run trains 7 baselines + 5 currency-pair ORS-FXNet models + 8
ablation variants + 5 walk-forward folds + 4 forecast horizons, so it can
take a long time on CPU. Run `--quick` first to make sure everything works,
then scale up.

## 5. Testing saved models

After `run_pipeline.py` has finished at least once, you can reload the
saved checkpoints and run inference on sample test data without retraining:

```bash
python test_saved_models.py                     # 10 sample rows per currency pair
python test_saved_models.py --n-samples 25
python test_saved_models.py --pairs USD_INR EUR_INR
python test_saved_models.py --device cpu
```

This prints an actual-vs-predicted table (in real exchange-rate units) for
each currency pair and writes it to
`outputs/tables/sample_test_predictions.csv`.

## 6. Where the outputs go

```
outputs/
├── figures/   all generated plots (.png)
├── tables/    all generated result tables (.csv)
├── models/    trained checkpoints:
│              ors_fxnet_<pair>.pt        neural base model per currency pair
│              rl_ensemble_<pair>.pt      DQN/DDPG/PPO correction ensemble per pair
│              scaler_bundle.pkl          fitted feature scalers
│              run_metadata.json          config needed to reload models later
└── logs/
    └── run_log.json                      run timing and CLI arguments used
```

## 7. Command reference

| Command | Purpose |
|---|---|
| `python run_pipeline.py` | Full training + evaluation run |
| `python run_pipeline.py --quick` | Fast end-to-end smoke test |
| `python run_pipeline.py --pairs USD EUR` | Restrict to specific currency pairs |
| `python run_pipeline.py --epochs N` | Override number of training epochs |
| `python run_pipeline.py --rl-episodes N` | Override number of RL training episodes |
| `python run_pipeline.py --device cpu\|cuda` | Force compute device |
| `python test_saved_models.py` | Run inference on saved models with sample test data |
| `python test_saved_models.py --n-samples N` | Number of sample rows shown per pair |
| `python test_saved_models.py --pairs <cols>` | Restrict testing to specific pair columns (e.g. `USD_INR`) |

## 8. Project layout

```
config.py                     all paths and hyperparameters
run_pipeline.py                main training/evaluation entry point
test_saved_models.py            reload saved checkpoints and run sample inference
requirements.txt                 Python dependencies
data/raw/                         input CSV files
outputs/                           all generated figures, tables, models, logs
src/
├── data_preparation.py         data loading, cleaning, feature engineering, splitting
├── datasets.py                  PyTorch Dataset / DataLoader wrappers
├── model_layers.py               core neural network layers
├── model.py                       full model assembly
├── rl_agents.py                    reinforcement-learning correction agents
├── baselines.py                     baseline forecasting models
├── train.py                          training loops
├── evaluation_metrics.py              evaluation metric functions
├── statistical_tests.py                statistical significance tests
├── ablation.py                          ablation-study runner
├── walk_forward.py                       walk-forward CV and multi-horizon forecasting
├── visualization_eda.py                   exploratory data plots
├── visualization_architecture.py           architecture diagram
└── visualization_results.py                 result/diagnostic plots
```

## 9. Troubleshooting

- **`FileNotFoundError` about missing CSVs** — place `Foreign_Exchange_Rates.csv`
  and `GoldUP.csv` in `data/raw/`.
- **Run is too slow** — use `--quick`, reduce `--epochs` / `--rl-episodes`,
  restrict `--pairs`, or use `--device cuda` on a machine with a GPU.
- **`outputs/models/run_metadata.json` not found when running
  `test_saved_models.py`** — run `python run_pipeline.py` first; it must
  complete at least once to produce saved checkpoints.
