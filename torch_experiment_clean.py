import os
import sys
import torch
import numpy as np
import pandas as pd
import time
import argparse
from functools import partial
from typing import Tuple, List, Dict, Optional

from metrics import top_k_onehot_indicator
from torch_perturb.perturbations import perturbed
from torch_models import NegativeBinomialRegressionModel, torch_bpr_uncurried, deterministic_bpr


def read_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load training and validation feature/target CSVs into DataFrames.

    Args:
        data_dir: Directory containing 'train_x.csv', 'train_y.csv', 'valid_x.csv', 'valid_y.csv'.

    Returns:
        Tuple of DataFrames: (train_X_df, train_Y_df, val_X_df, val_Y_df) with MultiIndex ['geoid','timestep'].
    """
    def _load(fname: str) -> pd.DataFrame:
        path = os.path.join(data_dir, fname)
        return pd.read_csv(path, index_col=[0, 1])

    train_X_df = _load('train_x.csv')
    train_Y_df = _load('train_y.csv')
    val_X_df   = _load('valid_x.csv')
    val_Y_df   = _load('valid_y.csv')
    return train_X_df, train_Y_df, val_X_df, val_Y_df


def convert_df_to_3d_array(df: pd.DataFrame) -> Tuple[np.ndarray, List, List]:
    """
    Convert a MultiIndex DataFrame with levels ['geoid','timestep'] to a 3D numpy array.

    Returns:
        X: shape (num_timesteps, num_locations, num_features)
        geoids: sorted list of geoids
        timesteps: sorted list of timesteps
    """
    if not isinstance(df.index, pd.MultiIndex) or set(df.index.names) != {'geoid', 'timestep'}:
        raise ValueError("DataFrame must have a MultiIndex with levels 'geoid' and 'timestep'")

    geoids = sorted(df.index.get_level_values('geoid').unique())
    timesteps = sorted(df.index.get_level_values('timestep').unique())
    geoid_to_idx = {g: i for i, g in enumerate(geoids)}

    T = len(timesteps)
    G = len(geoids)
    F = df.shape[1]
    X = np.zeros((T, G, F), dtype=float)

    for (geoid, timestep), row in df.iterrows():
        t_idx = timesteps.index(timestep)
        g_idx = geoid_to_idx[geoid]
        X[t_idx, g_idx, :] = row.values
    return X, geoids, timesteps


def convert_y_df_to_2d_array(y_df: pd.DataFrame, geoids: List, timesteps: List) -> np.ndarray:
    """
    Convert a MultiIndex target DataFrame to a 2D numpy array.

    Args:
        y_df: DataFrame with MultiIndex ['geoid','timestep'] and one column.
        geoids, timesteps: lists from convert_df_to_3d_array for consistent ordering.

    Returns:
        y: array shape (num_timesteps, num_locations)
    """
    if not isinstance(y_df.index, pd.MultiIndex) or set(y_df.index.names) != {'geoid', 'timestep'}:
        raise ValueError("DataFrame must have a MultiIndex with levels 'geoid' and 'timestep'")

    geoid_to_idx = {g: i for i, g in enumerate(geoids)}
    T = len(timesteps)
    G = len(geoids)
    y = np.zeros((T, G), dtype=float)

    for (geoid, timestep), value in y_df.iloc[:, 0].items():
        t_idx = timesteps.index(timestep)
        g_idx = geoid_to_idx[geoid]
        y[t_idx, g_idx] = value
    return y


def evaluate_model(model, X, y, time, K, M_score_func, perturbed_top_K_func) -> Dict[str, float]:
    """
    Evaluate model on given data and return metrics.

    Args:
        model: NegativeBinomialRegressionModel instance
        X: tensor of shape (T, G, F)
        y: tensor of shape (T, G)
        time: tensor of shape (T, G)
        K: top-K cutoff for BPR
        M_score_func: number of Monte Carlo samples
        perturbed_top_K_func: perturbed top-K function

    Returns:
        dict with keys 'nll', 'perturbed_bpr', 'deterministic_bpr'
    """
    model.eval()
    with torch.no_grad():
        dist = model(X, time)
        # Sample and calculate ratio ratings
        y_sample_TMS = dist.sample((M_score_func,)).permute(1, 0, 2)
        ratio_rating_TMS = y_sample_TMS / (1 + y_sample_TMS.sum(dim=-1, keepdim=True))
        ratio_rating_TS = ratio_rating_TMS.mean(dim=1)

        # Compute negative log-likelihood
        nll = -model.log_likelihood(y, X, time)
        # Compute BPR metrics
        perturbed_bpr_T = torch_bpr_uncurried(ratio_rating_TS, y, K=K,
                                              perturbed_top_K_func=perturbed_top_K_func)
        deterministic_bpr_T = deterministic_bpr(ratio_rating_TS, y, K=K)

        metrics = {
            'nll': nll.item(),
            'perturbed_bpr': torch.mean(perturbed_bpr_T).item(),
            'deterministic_bpr': torch.mean(deterministic_bpr_T).item()
        }
        return metrics


def train_epoch_neg_binom(model,
                           optimizer,
                           K,
                           epsilon,
                           M_score_func,
                           feat_TSF,
                           time_T,
                           train_y_TS,
                           perturbed_top_K_func,
                           bpr_weight,
                           nll_weight,
                           update: bool = True) -> Tuple[Dict[str, float], NegativeBinomialRegressionModel]:
    """
    Train one epoch of the negative binomial model.

    Returns:
        metrics dict and the updated model
    """
    model.train()
    optimizer.zero_grad()

    total_loss = 0.0
    total_gradient_P = None

    for t in range(feat_TSF.shape[0]):
        dist = model(feat_TSF[t:t+1], time_T[t:t+1])
        # Monte Carlo samples
        y_sample_TMS = dist.sample((M_score_func,)).permute(1, 0, 2)
        action_denominator_TM = y_sample_TMS.sum(dim=-1, keepdim=True) + 1
        ratio_rating_TMS = y_sample_TMS / action_denominator_TM
        ratio_rating_TS = ratio_rating_TMS.mean(dim=1)
        ratio_rating_TS.requires_grad_(True)

        # Score function estimator via Jacobian
        def get_log_probs_baked(param_tensor):
            dist_baked = model.build_from_single_tensor(param_tensor,
                                                        feat_TSF[t:t+1],
                                                        time_T[t:t+1])
            log_probs_TMS = dist_baked.log_prob(y_sample_TMS.permute(1, 0, 2))
            return log_probs_TMS.permute(1, 0, 2)

        jac_TMSP = torch.autograd.functional.jacobian(
            get_log_probs_baked,
            (model.params_to_single_tensor(),),
            strategy='forward-mode',
            vectorize=True
        )
        score_func_estimator_TMSP = jac_TMSP * ratio_rating_TMS.unsqueeze(-1)
        score_func_estimator_TSP = score_func_estimator_TMSP.mean(dim=1)

        # BPR loss
        positive_bpr_T = torch_bpr_uncurried(ratio_rating_TS,
                                             torch.tensor(train_y_TS[t:t+1]),
                                             K=K,
                                             perturbed_top_K_func=perturbed_top_K_func)
        if nll_weight > 0:
            bpr_threshold_diff_T = positive_bpr_T - epsilon
            violate_flag = bpr_threshold_diff_T < 0
            negative_bpr_loss = torch.mean(-bpr_threshold_diff_T * violate_flag)
        else:
            negative_bpr_loss = torch.mean(-positive_bpr_T)

        # NLL loss
        nll = -model.log_likelihood(train_y_TS[t:t+1], feat_TSF[t:t+1], time_T[t:t+1])
        loss = bpr_weight * negative_bpr_loss + nll_weight * nll
        loss.backward()

        # Accumulate parameter gradients
        loss_grad_TS = ratio_rating_TS.grad
        gradient_TSP = score_func_estimator_TSP * loss_grad_TS.unsqueeze(-1)
        gradient_P = torch.sum(gradient_TSP, dim=[0, 1])
        total_gradient_P = gradient_P if total_gradient_P is None else total_gradient_P + gradient_P
        total_loss += loss.item()

    # Unpack and apply gradients
    gradient_tuple = model.single_tensor_to_params(total_gradient_P)
    for param, grad in zip(model.parameters(), gradient_tuple):
        if nll_weight > 0:
            grad = grad + param.grad
        param.grad = grad

    if update:
        optimizer.step()

    # Final deterministic BPR on last batch
    deterministic_bpr_T = deterministic_bpr(ratio_rating_TS,
                                             torch.tensor(train_y_TS),
                                             K=K)
    det_bpr = torch.mean(deterministic_bpr_T).item()
    pert_bpr = torch.mean(positive_bpr_T).item()

    metrics = {
        'loss': total_loss,
        'deterministic_bpr': det_bpr,
        'perturbed_bpr': pert_bpr,
        'nll': nll.item()
    }
    return metrics, model


def main(K: Optional[int] = None,
         step_size: Optional[float] = None,
         epochs: Optional[int] = None,
         bpr_weight: Optional[float] = None,
         nll_weight: Optional[float] = None,
         seed: Optional[int] = None,
         outdir: Optional[str] = None,
         epsilon: Optional[float] = None,
         perturbed_noise: Optional[float] = None,
         num_score_samples: Optional[int] = None,
         num_pert_samples: Optional[int] = None,
         data_dir: Optional[str] = None,
         device: str = 'cuda',
         val_freq: int = 10):
    """
    Main training loop with command line arguments.
    """
    # Reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Load data
    train_X_df, train_Y_df, val_X_df, val_Y_df = read_data(data_dir)

    # Convert to arrays
    train_X, geoids, timesteps = convert_df_to_3d_array(train_X_df)
    train_time_arr = np.array([timesteps] * len(geoids)).T
    train_y = convert_y_df_to_2d_array(train_Y_df, geoids, timesteps)

    val_X, _, val_timesteps = convert_df_to_3d_array(val_X_df)
    val_time_arr = np.array([val_timesteps] * len(geoids)).T
    val_y = convert_y_df_to_2d_array(val_Y_df, geoids, val_timesteps)

    # Tensors on device
    X_train = torch.tensor(train_X, dtype=torch.float32).to(device)
    y_train = torch.tensor(train_y, dtype=torch.float32).to(device)
    time_train = torch.tensor(train_time_arr, dtype=torch.float32).to(device)

    X_val = torch.tensor(val_X, dtype=torch.float32).to(device)
    y_val = torch.tensor(val_y, dtype=torch.float32).to(device)
    time_val = torch.tensor(val_time_arr, dtype=torch.float32).to(device)

    # Model and optimizer
    model = NegativeBinomialRegressionModel(
        num_locations=len(geoids),
        num_fixed_effects=train_X.shape[2],
        device=device
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=step_size)

    # Top-K functions
    top_k_func = partial(top_k_onehot_indicator, k=K)
    perturbed_top_K_func = perturbed(top_k_func, sigma=perturbed_noise, num_samples=num_pert_samples)

    # Metrics storage
    metrics = {
        'train': {'epochs': [], 'loss': [], 'nll': [], 'perturbed_bpr': [], 'deterministic_bpr': []},
        'val':   {'epochs': [], 'nll': [], 'perturbed_bpr': [], 'deterministic_bpr': []},
        'times': []
    }
    best_val_loss = float('inf')

    for epoch in range(epochs):
        print(f'EPOCH: {epoch}')
        start = time.time()

        # Train epoch
        train_metrics, model = train_epoch_neg_binom(
            model, optimizer, K, epsilon,
            num_score_samples, X_train, time_train,
            y_train, perturbed_top_K_func,
            bpr_weight, nll_weight, update=True
        )
        # Record train metrics
        metrics['train']['epochs'].append(epoch)
        for k, v in train_metrics.items(): metrics['train'][k].append(v)

        # Validation
        if epoch % val_freq == 0:
            val_metrics = evaluate_model(
                model, X_val, y_val, time_val,
                K, num_score_samples, perturbed_top_K_func
            )
            metrics['val']['epochs'].append(epoch)
            for k, v in val_metrics.items(): metrics['val'][k].append(v)

            # Checkpoint best
            val_loss = val_metrics['nll'] * nll_weight - val_metrics['perturbed_bpr'] * bpr_weight
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(outdir, exist_ok=True)
                torch.save(model.state_dict(), f'{outdir}/best_model.pth')

        end = time.time()
        metrics['times'].append(end - start)

        print(f"Train - Loss: {train_metrics['loss']:.4f}, NLL: {train_metrics['nll']:.4f}, "
              f"BPR: {train_metrics['deterministic_bpr']:.4f}")
        if epoch % val_freq == 0:
            print(f"Val - NLL: {val_metrics['nll']:.4f}, "
                  f"BPR: {val_metrics['deterministic_bpr']:.4f}")

        # Periodic checkpoint
        if epoch % 100 == 0:
            os.makedirs(outdir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'best_val_loss': best_val_loss
            }, f'{outdir}/checkpoint.pth')

            # Save metrics CSVs
            pd.DataFrame(metrics['train']).set_index('epochs').to_csv(f'{outdir}/train_metrics.csv')
            pd.DataFrame(metrics['val']).set_index('epochs').to_csv(f'{outdir}/val_metrics.csv')
            pd.DataFrame({'times': metrics['times']}).to_csv(f'{outdir}/time_metrics.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=100)
    parser.add_argument("--step_size", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--bpr_weight", type=float, default=1.0)
    parser.add_argument("--nll_weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--epsilon", type=float, default=0.55)
    parser.add_argument("--perturbed_noise", type=float, default=0.01)
    parser.add_argument("--num_score_samples", type=int, default=20)
    parser.add_argument("--num_pert_samples", type=int, default=50)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--val_freq", type=int, default=10)

    args = parser.parse_args()
    main(**vars(args))
