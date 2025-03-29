import os
import torch
import numpy as np
from functools import partial
from torch.distributions import Normal
from torch_models import MixtureOfTruncNormModel, torch_bpr_uncurried, deterministic_bpr
from torch_perturb.perturbations import perturbed
from metrics import top_k_onehot_indicator

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_jacobian_all_together(model, y_sample_TMS, device):
    def get_log_probs_baked(param_P):
        distribution = model.build_from_single_tensor(param_P)
        return distribution.log_prob(y_sample_TMS)

    return torch.autograd.functional.jacobian(
        get_log_probs_baked,
        (model.params_to_single_tensor().to(device)),
        strategy='forward-mode',
        vectorize=True
    )

def compute_jacobian_per_parameter(model, y_sample_TMS, device):
    params = model.params_to_single_tensor().to(device)
    jacobians = []

    for i in range(len(params)):

        def single_param_log_prob(param_scalar):
            param_copy = params.clone()
            param_copy[i] = param_scalar
            distribution = model.build_from_single_tensor(param_copy)
            return distribution.log_prob(y_sample_TMS)

        jac = torch.autograd.functional.jacobian(
            single_param_log_prob,
            params[i],
            strategy='forward-mode',
            vectorize=True
        )
        jacobians.append(jac)

    return torch.stack(jacobians, dim=-1)

def train_epoch(model, optimizer, K, threshold, train_T, M_score_func, M_action, train_y_TS, perturbed_top_K_func, bpr_weight, nll_weight, device, compute_jacobian_together=True, update=True):
    optimizer.zero_grad()
    model = model.to(device)
    mix_model = model()

    # Sample data from the model
    y_sample_TMS = mix_model.sample((train_T, M_score_func)).to(device)
    y_sample_action_TMS = y_sample_TMS

    # Compute ratios
    ratio_rating_TMS = y_sample_action_TMS / y_sample_action_TMS.sum(dim=-1, keepdim=True)
    ratio_rating_TS = ratio_rating_TMS.mean(dim=1)
    ratio_rating_TS.requires_grad_(True)

    # Compute Jacobian using selected method
    if compute_jacobian_together:
        jac_TMSP = compute_jacobian_all_together(model, y_sample_TMS, device)
    else:
        jac_TMSP = compute_jacobian_per_parameter(model, y_sample_TMS, device)

    # Score function estimator
    score_func_estimator_TMSP = jac_TMSP * ratio_rating_TMS.unsqueeze(-1)
    score_func_estimator_TSP = score_func_estimator_TMSP.mean(dim=1)

    # BPR calculation
    positive_bpr_T = torch_bpr_uncurried(
        ratio_rating_TS, 
        torch.tensor(train_y_TS, device=device), 
        K=K, 
        perturbed_top_K_func=perturbed_top_K_func
    )

    if nll_weight > 0:
        bpr_threshold_diff_T = positive_bpr_T - threshold
        negative_bpr_loss = torch.mean(-bpr_threshold_diff_T * (bpr_threshold_diff_T < 0))
    else:
        negative_bpr_loss = torch.mean(-positive_bpr_T)

    nll = torch.mean(-mix_model.log_prob(torch.tensor(train_y_TS, device=device)))
    loss = bpr_weight * negative_bpr_loss + nll_weight * nll

    # Backpropagation
    loss.backward()

    if update:
        optimizer.step()

    return loss.detach(), torch.mean(positive_bpr_T).detach(), nll.detach(), mix_model

def main(T=1000, S=20, K=10, step_size=0.01, epochs=10, bpr_weight=1.0, nll_weight=1.0, seed=360, epsilon=0.55, num_components=4, perturbed_noise=0.05, perturbation_samples=100, score_func_samples=100, device='cuda', compute_jacobian_together=True):
    set_seed(seed)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    dist_S = [Normal(loc, 0.3) for loc in [10, 20, 30, 40, 50, 60, 100]]

    # Generate training data
    train_y_TS = np.zeros((T, S))
    for s in range(S):
        dist = dist_S[s % len(dist_S)]
        train_y_TS[:, s] = dist.sample((T,)).cpu().numpy()

    model = MixtureOfTruncNormModel(num_components=num_components, S=S, low=0, high=150).to(device)
    # print(num parameters)
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')
    optimizer = torch.optim.Adam(model.parameters(), lr=step_size)

    top_k_func = partial(top_k_onehot_indicator, k=K)
    perturbed_top_K_func = perturbed(top_k_func, sigma=perturbed_noise, num_samples=perturbation_samples, device=device)

    for epoch in range(epochs):
        print(f'EPOCH: {epoch}')
        loss, bpr, nll, _ = train_epoch(
            model,
            optimizer,
            K,
            epsilon,
            T,
            score_func_samples,
            score_func_samples,
            train_y_TS,
            perturbed_top_K_func,
            bpr_weight,
            nll_weight,
            device=device,
            compute_jacobian_together=compute_jacobian_together
        )

        print(f"Loss: {loss:.4f}, BPR: {bpr:.4f}, NLL: {nll:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default='cuda', help='Device to run on (cuda or cpu)')
    parser.add_argument("--compute_jacobian_together", action='store_true')
    args = parser.parse_args()

    print("\nRunning with Jacobian computed per parameter:")
    main(epochs=args.epochs, device=args.device, compute_jacobian_together=False, T=1000, S=1000, K=10)


    print("Running with Jacobian computed all together:")
    main(epochs=args.epochs, device=args.device, compute_jacobian_together=True, T=100, S=1000, K=10)

    

    
    
