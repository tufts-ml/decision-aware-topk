import os
import sys
import torch
import numpy as np
import pandas as pd
import time
import argparse
from functools import partial
from metrics import top_k_onehot_indicator
from torch_perturb.perturbations import perturbed
from torch_models import NegativeBinomialRegressionModel, torch_bpr_uncurried, deterministic_bpr


def convert_df_to_3d_array(df):
    # Ensure the DataFrame has a MultiIndex with 'geoid' and 'timestep'
    if not isinstance(df.index, pd.MultiIndex) or set(df.index.names) != {'geoid', 'timestep'}:
        raise ValueError("DataFrame must have a MultiIndex with levels 'geoid' and 'timestep'")

    # Get unique geoids and timesteps, sorted
    geoids = sorted(df.index.get_level_values('geoid').unique())
    timesteps = sorted(df.index.get_level_values('timestep').unique())

    # Create a mapping of geoids to indices
    geoid_to_idx = {geoid: idx for idx, geoid in enumerate(geoids)}

    # Initialize the 3D array
    num_timesteps = len(timesteps)
    num_locations = len(geoids)
    num_features = len(df.columns)
    X = np.zeros((num_timesteps, num_locations, num_features))

    # Fill the 3D array
    for (geoid, timestep), row in df.iterrows():
        t_idx = timesteps.index(timestep)
        g_idx = geoid_to_idx[geoid]
        X[t_idx, g_idx, :] = row.values

    return X, geoids, timesteps

def convert_y_df_to_2d_array(y_df, geoids, timesteps):
    # Ensure the DataFrame has a MultiIndex with 'geoid' and 'timestep'
    if not isinstance(y_df.index, pd.MultiIndex) or set(y_df.index.names) != {'geoid', 'timestep'}:
        raise ValueError("DataFrame must have a MultiIndex with levels 'geoid' and 'timestep'")

    # Initialize the 2D array
    num_timesteps = len(timesteps)
    num_locations = len(geoids)
    y = np.zeros((num_timesteps, num_locations))

    # Create a mapping of geoids to indices
    geoid_to_idx = {geoid: idx for idx, geoid in enumerate(geoids)}

    # Fill the 2D array
    for (geoid, timestep), value in y_df.iloc[:, 0].items():
        t_idx = timesteps.index(timestep)
        g_idx = geoid_to_idx[geoid]
        y[t_idx, g_idx] = value

    return y

def evaluate_model(model, X, y, time, K, M_score_func, perturbed_top_K_func):
    """Evaluate model on given data and return metrics."""
    with torch.no_grad():
        dist = model(X, time)
        
        # Sample and calculate ratio ratings
        y_sample_TMS = dist.sample((M_score_func,)).permute(1, 0, 2)
        ratio_rating_TMS = y_sample_TMS/(1+y_sample_TMS.sum(dim=-1, keepdim=True))
        ratio_rating_TS = ratio_rating_TMS.mean(dim=1)
        
        # Calculate metrics
        nll = -model.log_likelihood(y, X, time)
        perturbed_bpr_T = torch_bpr_uncurried(ratio_rating_TS, y, K=K, 
                                             perturbed_top_K_func=perturbed_top_K_func)
        deterministic_bpr_T = deterministic_bpr(ratio_rating_TS, y, K=K)
        
        metrics = {
            'nll': nll.item(),
            'perturbed_bpr': torch.mean(perturbed_bpr_T).item(),
            'deterministic_bpr': torch.mean(deterministic_bpr_T).item()
        }
        
        return metrics

def train_epoch_neg_binom(model, optimizer, K, epsilon,
                         M_score_func, feat_TSF,
                         time_T, train_y_TS,
                         perturbed_top_K_func, bpr_weight, nll_weight, update=True):
    """Train one epoch of the negative binomial model."""
    model.train()
    optimizer.zero_grad()
    
    total_loss = 0
    total_gradient_P = None
    
    for t in range(feat_TSF.shape[0]):
        dist = model(feat_TSF[t:t+1], time_T[t:t+1])
        
        y_sample_TMS = dist.sample((M_score_func,)).permute(1, 0, 2)
        y_sample_action_TMS = y_sample_TMS
        action_denominator_TM = y_sample_action_TMS.sum(dim=-1, keepdim=True) + 1 

        ratio_rating_TMS = y_sample_action_TMS / action_denominator_TM
        ratio_rating_TS = ratio_rating_TMS.mean(dim=1)
        ratio_rating_TS.requires_grad_(True)

        def get_log_probs_baked(param):
            distribution = model.build_from_single_tensor(param, feat_TSF[t:t+1], time_T[t:t+1])
            log_probs_TMS = distribution.log_prob(y_sample_TMS.permute(1, 0, 2)).permute(1, 0, 2)
            return log_probs_TMS

        jac_TMSP = torch.autograd.functional.jacobian(get_log_probs_baked, 
                                                      (model.params_to_single_tensor()), 
                                                      strategy='forward-mode', 
                                                      vectorize=True)

        score_func_estimator_TMSP = jac_TMSP * ratio_rating_TMS.unsqueeze(-1)
        score_func_estimator_TSP = score_func_estimator_TMSP.mean(dim=1)    

        positive_bpr_T = torch_bpr_uncurried(ratio_rating_TS, torch.tensor(train_y_TS[t:t+1]), 
                                             K=K, perturbed_top_K_func=perturbed_top_K_func)

        if nll_weight > 0:
            bpr_threshold_diff_T = positive_bpr_T - epsilon
            violate_threshold_flag = bpr_threshold_diff_T < 0
            negative_bpr_loss = torch.mean(-bpr_threshold_diff_T * violate_threshold_flag)
        else:
            negative_bpr_loss = torch.mean(-positive_bpr_T)

        nll = -model.log_likelihood(train_y_TS[t:t+1], feat_TSF[t:t+1], time_T[t:t+1])
        loss = bpr_weight * negative_bpr_loss + nll_weight * nll
        loss.backward()

        loss_grad_TS = ratio_rating_TS.grad
        gradient_TSP = score_func_estimator_TSP * torch.unsqueeze(loss_grad_TS, -1)
        gradient_P = torch.sum(gradient_TSP, dim=[0, 1])
        
        if total_gradient_P is None:
            total_gradient_P = gradient_P
        else:
            total_gradient_P += gradient_P
        
        total_loss += loss.item()

    gradient_tuple = model.single_tensor_to_params(total_gradient_P)

    for param, gradient in zip(model.parameters(), gradient_tuple):
        if nll_weight > 0:
            gradient = gradient + param.grad
        param.grad = gradient

    if update:
        optimizer.step()

    deterministic_bpr_T = deterministic_bpr(ratio_rating_TS, torch.tensor(train_y_TS), K=K)
    det_bpr = torch.mean(deterministic_bpr_T)

    metrics = {
        'loss': total_loss ,
        'deterministic_bpr': det_bpr.item(),
        'perturbed_bpr': torch.mean(positive_bpr_T).item(),
        'nll': nll.item()
    }

    return metrics, model

def main(K=None, step_size=None, epochs=None, bpr_weight=None,
         nll_weight=None, seed=None, outdir=None, epsilon=None,
         perturbed_noise=None, num_score_samples=None, num_pert_samples=None,
         data_dir=None, device='cuda', val_freq=10):
    """Main training loop with command line arguments."""
    
    # Set random seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Load training data
    train_X_df = pd.read_csv(os.path.join(data_dir, 'train_x.csv'), index_col=[0,1])
    train_Y_df = pd.read_csv(os.path.join(data_dir, 'train_y.csv'), index_col=[0,1])
    
    # Load validation data
    val_X_df = pd.read_csv(os.path.join(data_dir, 'valid_x.csv'), index_col=[0,1])
    val_Y_df = pd.read_csv(os.path.join(data_dir, 'valid_y.csv'), index_col=[0,1])
    
    # Process training data
    train_X, geoids, timesteps = convert_df_to_3d_array(train_X_df)#.drop(columns='timestep.1'))
    train_time_arr = np.array([timesteps] * len(geoids)).T
    train_y = convert_y_df_to_2d_array(train_Y_df, geoids, timesteps)

    # Process validation data
    val_X, _, val_timesteps = convert_df_to_3d_array(val_X_df)#.drop(columns='timestep.1'))
    val_time_arr = np.array([val_timesteps] * len(geoids)).T
    val_y = convert_y_df_to_2d_array(val_Y_df, geoids, val_timesteps)

    # Convert to tensors and move to device
    X_train = torch.tensor(train_X, dtype=torch.float32).to(device)
    y_train = torch.tensor(train_y, dtype=torch.float32).to(device)
    time_train = torch.tensor(train_time_arr, dtype=torch.float32).to(device)
    
    X_val = torch.tensor(val_X, dtype=torch.float32).to(device)
    y_val = torch.tensor(val_y, dtype=torch.float32).to(device)
    time_val = torch.tensor(val_time_arr, dtype=torch.float32).to(device)

    # Initialize model
    model = NegativeBinomialRegressionModel(
        num_locations=len(geoids),
        num_fixed_effects=train_X.shape[2], device=device
    ).to(device)

    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=step_size)

    # Setup top-k function
    top_k_func = partial(top_k_onehot_indicator, k=K)
    perturbed_top_K_func = perturbed(top_k_func, sigma=perturbed_noise, num_samples=num_pert_samples)

    # Initialize metric tracking with separate epoch tracking for validation
    metrics = {
        'train': {
            'epochs': [], 
            'loss': [], 
            'nll': [], 
            'perturbed_bpr': [], 
            'deterministic_bpr': []
        },
        'val': {
            'epochs': [], 
            'nll': [], 
            'perturbed_bpr': [], 
            'deterministic_bpr': []
        },
        'times': []
    }

    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(epochs):
        print(f'EPOCH: {epoch}')
        start = time.time()
        
        # Training step
        train_metrics, model = train_epoch_neg_binom(
            model, optimizer, K, epsilon,
            num_score_samples, X_train, time_train,
            y_train, perturbed_top_K_func,
            bpr_weight, nll_weight, device
        )
        
        # Update training metrics
        metrics['train']['epochs'].append(epoch)
        for metric, value in train_metrics.items():
            metrics['train'][metric].append(value)
        
        # Validation step (every val_freq epochs)
        if epoch % val_freq == 0:
            model.eval()
            val_metrics = evaluate_model(
                model, X_val, y_val, time_val,
                K, num_score_samples, perturbed_top_K_func
            )
            
            # Update validation metrics
            metrics['val']['epochs'].append(epoch)
            for metric, value in val_metrics.items():
                metrics['val'][metric].append(value)
            
            # Save best model
            val_loss = val_metrics['nll'] * nll_weight
            if bpr_weight > 0:
                val_loss -= val_metrics['perturbed_bpr'] * bpr_weight
                
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if not os.path.exists(outdir):
                    os.makedirs(outdir)
                torch.save(model.state_dict(), f'{outdir}/best_model.pth')
        
        end = time.time()
        metrics['times'].append(end - start)
        
        # Print progress
        print(f"Train - Loss: {train_metrics['loss']:.4f}, NLL: {train_metrics['nll']:.4f}, "
              f"BPR: {train_metrics['deterministic_bpr']:.4f}")
        if epoch % val_freq == 0:
            print(f"Val - NLL: {val_metrics['nll']:.4f}, "
                  f"BPR: {val_metrics['deterministic_bpr']:.4f}")
        
        # Save checkpoints
        if epoch % 100 == 0:
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'best_val_loss': best_val_loss
            }, f'{outdir}/checkpoint.pth')
            
            # Save metrics separately for easier analysis
            # Create DataFrames with proper indexing
            train_df = pd.DataFrame(metrics['train']).set_index('epochs')
            val_df = pd.DataFrame(metrics['val']).set_index('epochs')
            times_df = pd.DataFrame({'times': metrics['times']}, index=range(len(metrics['times'])))
            
            train_df.to_csv(f'{outdir}/train_metrics.csv')
            val_df.to_csv(f'{outdir}/val_metrics.csv')
            times_df.to_csv(f'{outdir}/time_metrics.csv')

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