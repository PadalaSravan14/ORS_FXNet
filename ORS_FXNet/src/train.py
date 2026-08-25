
import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.model_layers import reconstruction_loss, sensitivity_loss
from src.rl_agents import MultiAgentCorrectionEnsemble, build_state_vector, correction_reward


def huber_loss(y_pred, y_true, delta=1.0):
    return F.huber_loss(y_pred, y_true, delta=delta)  # Eq. (15)


def train_neural_forecaster(model, train_loader, val_loader, pair, config,
                             short_window, long_window, macro_idx, tech_idx,
                             device="cpu", verbose=True):

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.base_learning_rate)
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(config.epochs):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            seq = batch["sequence"].to(device)
            target = batch["target"].to(device)

            x_short = seq[:, -short_window:, :]
            x_long = seq[:, -min(long_window, seq.size(1)):, :]
            macro_vec = seq[:, -1, macro_idx] if macro_idx else torch.zeros(seq.size(0), 1, device=device)
            tech_vec = seq[:, -1, tech_idx] if tech_idx else torch.zeros(seq.size(0), 1, device=device)

            outputs = model(x_short, x_long, macro_vec, tech_vec, pair)
            loss = huber_loss(outputs["y_base"], target, delta=config.huber_delta)

            if outputs["macro_reconstruction"] is not None:
                rec_loss = reconstruction_loss(outputs["macro_input"], outputs["macro_reconstruction"])
                try:
                    sens_loss = sensitivity_loss(outputs["y_base"], outputs["macro_latent"])
                except RuntimeError:
                    sens_loss = torch.tensor(0.0, device=device)
                loss = loss + rec_loss + 0.01 * sens_loss  # Eq. (23)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                seq = batch["sequence"].to(device)
                target = batch["target"].to(device)
                x_short = seq[:, -short_window:, :]
                x_long = seq[:, -min(long_window, seq.size(1)):, :]
                macro_vec = seq[:, -1, macro_idx] if macro_idx else torch.zeros(seq.size(0), 1, device=device)
                tech_vec = seq[:, -1, tech_idx] if tech_idx else torch.zeros(seq.size(0), 1, device=device)
                outputs = model(x_short, x_long, macro_vec, tech_vec, pair)
                val_losses.append(huber_loss(outputs["y_base"], target, delta=config.huber_delta).item())

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if verbose:
            print(f"[{pair}] epoch {epoch + 1}/{config.epochs} - train {train_loss:.5f} - val {val_loss:.5f}")

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                if verbose:
                    print(f"[{pair}] early stopping at epoch {epoch + 1}")
                break

    model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def generate_base_predictions(model, loader, pair, short_window, long_window,
                               macro_idx, tech_idx, device="cpu"):
    model.eval()
    preds, targets, macro_summaries, temporal_summaries = [], [], [], []
    for batch in loader:
        seq = batch["sequence"].to(device)
        target = batch["target"].to(device)
        x_short = seq[:, -short_window:, :]
        x_long = seq[:, -min(long_window, seq.size(1)):, :]
        macro_vec = seq[:, -1, macro_idx] if macro_idx else torch.zeros(seq.size(0), 1, device=device)
        tech_vec = seq[:, -1, tech_idx] if tech_idx else torch.zeros(seq.size(0), 1, device=device)
        outputs = model(x_short, x_long, macro_vec, tech_vec, pair)
        preds.append(outputs["y_base"].cpu().numpy())
        targets.append(target.cpu().numpy())
        temporal_summaries.append(outputs["fused_features"].cpu().numpy())
        macro_summaries.append(outputs["macro_latent"].cpu().numpy() if outputs["macro_latent"].numel() else
                                np.zeros((seq.size(0), 1)))
    return (np.concatenate(preds), np.concatenate(targets),
            np.concatenate(temporal_summaries), np.concatenate(macro_summaries))


def train_correction_ensemble(y_base_val, y_true_val, temporal_summary_val, macro_summary_val,
                               dqn_cfg, ddpg_cfg, ppo_cfg, episodes=500, episode_length=20,
                               device="cpu", verbose=True):
 
    ensemble = MultiAgentCorrectionEnsemble(dqn_cfg, ddpg_cfg, ppo_cfg, device)
    n = len(y_base_val)
    state_dim = dqn_cfg.state_dim
    rolling_vol = _rolling_volatility(y_true_val)

    reward_history = []
    for episode in range(episodes):
        start = np.random.randint(0, max(1, n - episode_length))
        prev_error = 0.0
        episode_reward = 0.0
        for t in range(start, min(start + episode_length, n - 1)):
            h_summary = np.mean(temporal_summary_val[t]) if temporal_summary_val[t].size else 0.0
            z_summary = np.mean(macro_summary_val[t]) if macro_summary_val[t].size else 0.0
            state = build_state_vector(y_base_val[t], h_summary, z_summary, rolling_vol[t], prev_error, state_dim)
            state_t = torch.as_tensor(state, dtype=torch.float32)

            dqn_idx, delta_dqn = ensemble.dqn_agent.select_action(state_t)
            delta_ddpg = ensemble.ddpg_agent.select_action(state_t)
            w1, log_prob, value = ensemble.ppo_agent.select_action(state_t)
            w2 = 1.0 - w1
            y_final = y_base_val[t] + w1 * delta_dqn + w2 * delta_ddpg

            y_prev_true = y_true_val[t - 1] if t > 0 else y_true_val[t]
            reward = correction_reward(y_final, y_true_val[t], y_prev_true)
            episode_reward += reward

            next_h = np.mean(temporal_summary_val[t + 1]) if temporal_summary_val[t + 1].size else 0.0
            next_z = np.mean(macro_summary_val[t + 1]) if macro_summary_val[t + 1].size else 0.0
            next_error = y_true_val[t] - y_final
            next_state = build_state_vector(y_base_val[t + 1], next_h, next_z, rolling_vol[t + 1], next_error, state_dim)
            next_state_t = torch.as_tensor(next_state, dtype=torch.float32)

            done = float(t == min(start + episode_length, n - 1) - 1)
            ensemble.dqn_agent.store(state_t, dqn_idx, reward, next_state_t, done)
            ensemble.ddpg_agent.store(state_t, delta_ddpg, reward, next_state_t, done)
            ensemble.ppo_agent.store(state_t, w1, log_prob, reward, value, done)

            prev_error = next_error

        ensemble.dqn_agent.update()
        ensemble.ddpg_agent.update()
        ensemble.dqn_agent.decay_epsilon()
        if (episode + 1) % 5 == 0:
            ensemble.dqn_agent.sync_target()
            ensemble.ppo_agent.update()

        reward_history.append(episode_reward)
        if verbose and (episode + 1) % max(1, episodes // 10) == 0:
            print(f"RL episode {episode + 1}/{episodes} - reward {episode_reward:.4f}")

    ensemble.ppo_agent.update()  # flush any remaining trajectory
    return ensemble, reward_history


def _rolling_volatility(series, window=21):
    series = np.asarray(series)
    vol = np.zeros_like(series)
    for i in range(len(series)):
        start = max(0, i - window + 1)
        w = series[start:i + 1]
        vol[i] = np.std(w) if len(w) > 1 else 1e-4
    return vol


@torch.no_grad()
def apply_correction_ensemble(ensemble, y_base, y_true, temporal_summary, macro_summary, device="cpu"):
    
    n = len(y_base)
    rolling_vol = _rolling_volatility(y_true)
    y_final = np.zeros(n)
    prev_error = 0.0
    for t in range(n):
        h_summary = np.mean(temporal_summary[t]) if temporal_summary[t].size else 0.0
        z_summary = np.mean(macro_summary[t]) if macro_summary[t].size else 0.0
        state = build_state_vector(y_base[t], h_summary, z_summary, rolling_vol[t], prev_error, ensemble.dqn_agent.cfg.state_dim)
        y_hat, info = ensemble.correct(state, y_base[t], greedy=True)
        y_final[t] = y_hat
        prev_error = y_true[t] - y_hat
    return y_final
