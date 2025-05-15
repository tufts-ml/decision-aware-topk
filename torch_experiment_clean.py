import os
import argparse
import time
from functools import partial
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import torch

from metrics import top_k_onehot_indicator
from torch_perturb.perturbations import perturbed
from torch_models import NegativeBinomialRegressionModel, torch_bpr_uncurried, deterministic_bpr


def read_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Read training and validation data from CSV files.

    Args:
        data_dir: Path to the directory containing train_x.csv, train_y.csv,
                  valid_x.csv, and valid_y.csv.

    Returns:
        train_X_df: Features for training (MultiIndex: geoid, timestep).
        train_Y_df: Targets for training (MultiIndex: geoid, timestep).
        val_X_df:   Features for validation (MultiIndex: geoid, timestep).
        val_Y_df:   Targets for validation (MultiIndex: geoid, timestep).
    """
    def _load(fname: str) -> pd.DataFrame:
        path = os.path.join(data_dir, fname)
        return pd.read_csv(path, index_col=[0, 1])

    train_X_df = _load('train_x.csv')
    train_Y_df = _load('train_y.csv')
    val_X_df   = _load('valid_x.csv')
    val_Y_df   = _load('valid_y.csv')
    return train_X_df, train_Y_df, val_X_df, val_Y_df


def df_to_3d_array(df: pd.DataFrame) -> Tuple[np.ndarray, List, List]:
    """
    Convert a MultiIndex DataFrame (geoid, timestep) into a 3D numpy array.

    Returns:
        X: Array of shape (num_timesteps, num_locations, num_features).
        geoids: Sorted list of geoid labels.
        timesteps: Sorted list of timesteps.
    """
    if not isinstance(df.index, pd.MultiIndex) or set(df.index.names) != {'geoid', 'timestep'}:
        raise ValueError("DataFrame must have MultiIndex with levels ('geoid','timestep')")

    geoids = sorted(df.index.get_level_values('geoid').unique())
    timesteps = sorted(df.index.get_level_values('timestep').unique())
    idx_map = {g: i for i, g in enumerate(geoids)}

    T, G, F = len(timesteps), len(geoids), df.shape[1]
    X = np.zeros((T, G, F), dtype=float)

    for (g, t), row in df.iterrows():
        X[timesteps.index(t), idx_map[g], :] = row.values
    return X, geoids, timesteps


def y_df_to_2d_array(y_df: pd.DataFrame, geoids: List, timesteps: List) -> np.ndarray:
    """
    Convert a MultiIndex target DataFrame into a 2D numpy array.

    Args:
        y_df: MultiIndex DataFrame with one column.
        geoids, timesteps: Lists from df_to_3d_array for consistent ordering.

    Returns:
        y: Array of shape (num_timesteps, num_locations).
    """
    if not isinstance(y_df.index, pd.MultiIndex) or set(y_df.index.names) != {'geoid', 'timestep'}:
        raise ValueError("DataFrame must have MultiIndex with levels ('geoid','timestep')")

    idx_map = {g: i for i, g in enumerate(geoids)}
    T, G = len(timesteps), len(geoids)
    y = np.zeros((T, G), dtype=float)

    for (g, t), val in y_df.iloc[:, 0].items():
        y[timesteps.index(t), idx_map[g]] = val
    return y


def evaluate_model(
    model: NegativeBinomialRegressionModel,
    X: torch.Tensor,
    y: torch.Tensor,
    time_tensor: torch.Tensor,
    K: int,
    num_samples: int,
    perturbed_topk: partial
) -> Dict[str, float]:
    """
    Compute metrics (NLL, perturbed BPR, deterministic BPR) for a given model.
    """
    model.eval()
    with torch.no_grad():
        dist = model(X, time_tensor)
        samples = dist.sample((num_samples,)).permute(1, 0, 2)
        ratios = samples / (1 + samples.sum(dim=-1, keepdim=True))
        mean_ratios = ratios.mean(dim=1)

        nll = -model.log_likelihood(y, X, time_tensor)
        pert_bpr = torch.mean(torch_bpr_uncurried(mean_ratios, y, K=K,
                                                 perturbed_top_K_func=perturbed_topk))
        det_bpr = torch.mean(deterministic_bpr(mean_ratios, y, K=K))

    return {'nll': nll.item(), 'perturbed_bpr': pert_bpr.item(), 'deterministic_bpr': det_bpr.item()}


def train_epoch(
    model,
    optimizer: torch.optim.Optimizer,
    X: torch.Tensor,
    y: torch.Tensor,
    time_tensor: torch.Tensor,
    K: int,
    epsilon: float,
    num_score_samples: int,
    perturbed_topk: partial,
    bpr_weight: float,
    nll_weight: float,
    update: bool = True
):
    """
    Train one epoch of the negative binomial model using BPR and NLL losses.
    """
    model.train()
    optimizer.zero_grad()
    total_loss = 0.0

    for t in range(X.shape[0]):
        dist = model(X[t:t+1], time_tensor[t:t+1])
        samples = dist.sample((num_score_samples,)).permute(1, 0, 2)
        ratios = samples / (1 + samples.sum(dim=-1, keepdim=True))
        mean_ratios = ratios.mean(dim=1)

        # compute BPR loss
        pos_bpr = torch_bpr_uncurried(mean_ratios, y[t:t+1], K=K,
                                      perturbed_top_K_func=perturbed_topk)
        if nll_weight > 0:
            margin = pos_bpr - epsilon
            neg_bpr_loss = torch.mean(torch.clamp(-margin, min=0.0))
        else:
            neg_bpr_loss = torch.mean(-pos_bpr)

        # compute NLL loss
        nll = -model.log_likelihood(y[t:t+1], X[t:t+1], time_tensor[t:t+1])
        loss = bpr_weight * neg_bpr_loss + nll_weight * nll
        loss.backward()
        total_loss += loss.item()

    if update:
        optimizer.step()

    # final metrics for epoch
    det_bpr = torch.mean(deterministic_bpr(mean_ratios, y, K=K)).item()
    pert_bpr = torch.mean(pos_bpr).item()
    return (
        {'loss': total_loss, 'nll': nll.item(), 'perturbed_bpr': pert_bpr, 'deterministic_bpr': det_bpr},
        model
    )


def main():
    """
    Parse arguments, load data, prepare tensors, run training and evaluation loop.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Directory with CSV data files')
    parser.add_argument('--outdir', required=True, help='Output directory for models and metrics')
    parser.add_argument('--K', type=int, default=100)
    parser.add_argument('--step_size', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--bpr_weight', type=float, default=1.0)
    parser.add_argument('--nll_weight', type=float, default=1.0)
    parser.add_argument('--epsilon', type=float, default=0.55)
    parser.add_argument('--perturbed_noise', type=float, default=0.01)
    parser.add_argument('--num_score_samples', type=int, default=20)
    parser.add_argument('--num_pert_samples', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--val_freq', type=int, default=10)

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # load raw DataFrames
    train_X_df, train_Y_df, val_X_df, val_Y_df = read_data(args.data_dir)

    # convert to arrays
    X_train, geoids, times = df_to_3d_array(train_X_df)
    y_train = y_df_to_2d_array(train_Y_df, geoids, times)
    X_val, _, _ = df_to_3d_array(val_X_df)
    y_val = y_df_to_2d_array(val_Y_df, geoids, times)

    # prepare tensors
    device = torch.device(args.device)
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    times_t   = torch.tensor(np.array([times]*len(geoids)).T, dtype=torch.float32, device=device)
    X_val_t   = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t   = torch.tensor(y_val, dtype=torch.float32, device=device)
    times_val = times_t.clone()

    # model, optimizer, and top-k functions
    model = NegativeBinomialRegressionModel(
        num_locations=len(geoids), num_fixed_effects=X_train.shape[2], device=device
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.step_size)
    topk = partial(top_k_onehot_indicator, k=args.K)
    perturbed_topk = perturbed(topk, sigma=args.perturbed_noise, num_samples=args.num_pert_samples)

    best_val = float('inf')
    for epoch in range(args.epochs):
        start = time.time()
        train_metrics, model = train_epoch(
            model, optimizer, X_train_t, y_train_t, times_t,
            args.K, args.epsilon, args.num_score_samples,
            perturbed_topk, args.bpr_weight, args.nll_weight
        )

        if epoch % args.val_freq == 0:
            val_metrics = evaluate_model(
                model, X_val_t, y_val_t, times_val,
                args.K, args.num_score_samples, perturbed_topk
            )
            # logic to save best model omitted for brevity

        end = time.time()
        print(f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f} | "
              f"val_nll={val_metrics['nll']:.4f} if computed | took {end-start:.2f}s")

    # final save and metrics export can be added here

if __name__ == '__main__':
    main()
