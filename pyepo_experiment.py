import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import pyepo
import time
import argparse
from torch_models import PyEPONegativeBinomialRegressionModel, deterministic_bpr
from data_loader import load_data


def run_epoch_pyepo(
    prediction_model,
    loader,
    M_score_func,
    K,
    loss_func,
    method_name,
    optmodel=None,            # make optional in case we only do evaluation
    optimizer=None,          # make optional in case we only do evaluation
    device='cpu',
    train=True,               # flag to set training or evaluation mode
    scale_truth=False 
):
    """
    Run one "epoch" of either training or evaluation.

    :param prediction_model: model used for predictions
    :param loader: dataloader (train, valid, or test)
    :param M_score_func: integer or function for ratio_rating
    :param K: K for deterministic_bpr
    :param loss_func: the loss function
    :param method_name: which method to run
    :param optimizer: optimizer used in training (if any)
    :param device: 'cpu' or 'cuda'
    :param train: whether this pass should perform training (True) or evaluation (False)

    :return: (prediction_model, metrics) 
    """
    if train:
        prediction_model.train()
    else:
        prediction_model.eval()

    total_loss = 0.0
    total_bpr = []
    total_nll = []
    total_regret = []

    # If we're evaluating, we don't need gradients
    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for i, data in enumerate(loader):
            x, c, w, z = data
            if device == 'cuda':
                x, c, w, z = x.cuda(), c.cuda(), w.cuda(), z.cuda()

            # forward pass
            cp = prediction_model.ratio_rating(x, num_samples=M_score_func)

            if scale_truth:
                # divide each batch by the sum of the batch
                c_scaled = c / c.sum(dim=1, keepdim=True)
            else:
                c_scaled = c
            # compute loss
            if method_name == "spo+":
                loss = loss_func(cp, c_scaled, w, z)
            elif method_name in ["ptb", "pfy", "imle", "nce"]:
                loss = loss_func(cp, w)
            elif method_name in ["dbb", "nid"]:
                loss = loss_func(cp, c_scaled, z)
            elif method_name in ["2s", "pg", "ltr"]:
                loss = loss_func(cp, c_scaled)
            else:
                raise ValueError(f"Unknown method {method_name}")

            # only backprop and optimize if training
            if train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # compute BPR and NLL
            bpr = deterministic_bpr(cp, c, K=K)
            nll = -prediction_model.log_likelihood(c, x)
            regret  = pyepo.metric.regret(prediction_model, optmodel, loader)

            total_loss += loss.item()
            total_bpr.append(bpr)
            total_nll.append(nll)
            total_regret.append(regret)

    # compute means
    avg_loss = total_loss / len(loader)
    all_bpr = torch.cat([bpr.flatten() for bpr in total_bpr])
    avg_bpr = all_bpr.mean().detach().cpu().numpy()

    all_nll = torch.cat([nll.flatten() for nll in total_nll])
    avg_nll = all_nll.mean().detach().cpu().numpy()
    avg_regret = np.mean(total_regret)


    metrics = {
        'loss': avg_loss,
        'deterministic_bpr': avg_bpr,
        'nll': avg_nll,
        'regret': avg_regret
    }

    return metrics, prediction_model


def main(K=None, step_size=None, epochs=None,  seed=None, outdir=None, num_score_samples=None,
         method_name=None, pg_sigma=None,
         data_dir=None, batch_size=4, device='cuda', val_freq=10):
    """Main training loop with command line arguments."""
    
    # Set random seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    X_train, y_train, train_time = load_data(data_dir, 'train', device='cpu')
    # add train_time [B,S] to the end of X_train [B,S,F]
    X_train = torch.cat([X_train, train_time.unsqueeze(-1)], dim=-1)
    capacities = [K]
    num_locations = X_train.shape[1]
    weights = np.ones((1, num_locations))
    optmodel = pyepo.model.grb.knapsackModel(weights, capacities)
    # get training data set
    dataset_train = pyepo.data.dataset.optDataset(optmodel, X_train, y_train)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    X_val, y_val, val_time = load_data(data_dir, 'valid', device='cpu')
    X_val = torch.cat([X_val, val_time.unsqueeze(-1)], dim=-1)
    dataset_val = pyepo.data.dataset.optDataset(optmodel, X_val, y_val)
    loader_val = DataLoader(dataset_val, batch_size=len(dataset_val), shuffle=False)

    # Initialize model
    model = PyEPONegativeBinomialRegressionModel(
        num_locations=X_train.shape[1],
        num_fixed_effects=X_train.shape[2]-1, device=device
    ).to(device)

    if method_name =='spo+':
        spop = pyepo.func.SPOPlus(optmodel, processes=1)
        loss_func = spop
    elif method_name == 'pg':
        loss_func = pyepo.func.perturbationGradient(optmodel, sigma=pg_sigma, two_sides=False, processes=1)
    elif method_name == 'ltr':
        loss_func = pyepo.func.LTR(optmodel, processes=1)
    else:
        raise ValueError(f"Unknown method {method_name}")

    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=step_size)

      # Initialize metric tracking with separate epoch tracking for validation
    metrics = {
        'train': {
            'epochs': [], 
            'loss': [], 
            'nll': [], 
            'deterministic_bpr': [],
            'regret': [],
        },
        'val': {
            'epochs': [], 
            'loss': [], 
            'nll': [], 
            'deterministic_bpr': [],
            'regret': [],
        },
        'times': []
    }

    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(epochs):
        print(f'EPOCH: {epoch}')
        start = time.time()
        
        train_metrics, model = run_epoch_pyepo(
            model,
            loader_train,
            num_score_samples,
            K,
            loss_func,
            method_name,
            optmodel=optmodel,
            optimizer=optimizer,          
            device=device,
            train=True           
        )
        
        # Update training metrics
        metrics['train']['epochs'].append(epoch)
        for metric, value in train_metrics.items():
            metrics['train'][metric].append(value)
        
        # Validation step (every val_freq epochs)
        if epoch % val_freq == 0:
            val_metrics, model = run_epoch_pyepo(
                model,
                loader_val,
                num_score_samples,
                K,
                loss_func,
                method_name,
                optmodel=optmodel,
                optimizer=None,          
                device=device,
                train=False           
            )
            
            # Update validation metrics
            metrics['val']['epochs'].append(epoch)
            for metric, value in val_metrics.items():
                metrics['val'][metric].append(value)
            
            # Save best model
            val_loss = val_metrics['loss']

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if not os.path.exists(outdir):
                    os.makedirs(outdir)
                torch.save(model.state_dict(), f'{outdir}/best_model.pth')
        
        end = time.time()
        metrics['times'].append(end - start)
        
        # Print progress
        print(f"Train - Loss: {train_metrics['loss']:.4f}, NLL: {train_metrics['nll']:.4f}, "
              f"BPR: {train_metrics['deterministic_bpr']:.4f}, Regret: {train_metrics['regret']:.4f}")
        if epoch % val_freq == 0:
            print(f"Val - Loss: {val_metrics['loss']:.4f}, NLL: {val_metrics['nll']:.4f}, "
                  f"BPR: {val_metrics['deterministic_bpr']:.4f}, Regret: {val_metrics['regret']:.4f}")
        
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
            # set all tensors to cpu
            for key in metrics.keys():
                if key == 'times':
                    continue
                for metric in metrics[key].keys():
                    if metric == 'epochs' or metric =='loss':
                        continue
                    try:
                        metrics[key][metric] = [val.detach().cpu().numpy() for val in metrics[key][metric]]
                    except:
                        # whatever man
                        pass
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--num_score_samples", type=int, default=20)
    parser.add_argument("--dataset", type=str, choices=['asurv', 'cook', 'MA'])
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--val_freq", type=int, default=10)
    parser.add_argument("--method_name", type=str, required=True)
    parser.add_argument("--pg_sigma", type=float, default=1.0)

    ma_data_dir = '/cluster/tufts/hugheslab/datasets/NSF_OD/cleaned/long/MA/'
    cook_data_dir = '/cluster/tufts/hugheslab/kheuto01/code/decision-aware-topk/data/cook_county/'
    asurv_data_dir = '/cluster/tufts/hugheslab/kheuto01/code/decision-aware-topk/data/aerial_surv/'
    args = parser.parse_args()
    if args.dataset == 'asurv':
        data_dir = asurv_data_dir
    elif args.dataset == 'cook':
        data_dir = cook_data_dir
    elif args.dataset == 'MA':
        data_dir = ma_data_dir

    # remove dataset from args
    args = vars(args)
    args.pop('dataset')
    main(data_dir=data_dir, **args)