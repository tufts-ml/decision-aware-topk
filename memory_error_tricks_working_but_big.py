import os
import torch
import numpy as np
import gc
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

def compute_jacobian_accumulating(model, y_sample_TMS, device, batch_size_T=10, batch_size_M=5):
    """Compute Jacobian without storing the full tensor, batching over both T and M."""
    with torch.no_grad():
        params = model.params_to_single_tensor().detach().to(device)
        y_sample_detached = y_sample_TMS.detach()
    
    T, M, S = y_sample_TMS.shape
    P = params.shape[0]

    def get_log_probs(param_P, batch_samples):
        """Helper function to compute log probabilities."""
        with torch.set_grad_enabled(True):
            distribution = model.build_from_single_tensor(param_P)
            return distribution.log_prob(batch_samples)
    with torch.no_grad():
        accumulated_score = torch.zeros((T, S, P), device=device)  # Store reduced output only

        for t_start in range(0, T, batch_size_T):
            t_end = min(t_start + batch_size_T, T)

            for m_start in range(0, M, batch_size_M):
                m_end = min(m_start + batch_size_M, M)

                batch_samples = y_sample_detached[t_start:t_end, m_start:m_end]  # Slice over both T and M
                print(batch_samples.shape)

                def single_param_log_prob(param_P):
                    return get_log_probs(param_P, batch_samples)

                # Compute Jacobian for this batch
                batch_jacobian = torch.autograd.functional.jacobian(
                    single_param_log_prob,
                    params,
                    strategy='forward-mode',
                    vectorize=True
                )

                # Aggregate result without storing large tensor
                accumulated_score[t_start:t_end] += batch_jacobian.mean(dim=1) / (M // batch_size_M)

                del batch_jacobian
                torch.cuda.empty_cache()
                gc.collect()

    return accumulated_score



def compute_jacobian_all_together_batched(model, y_sample_TMS, device, batch_size=10):
    """Compute Jacobian in batches to reduce memory usage"""
    T, M, S = y_sample_TMS.shape
    
    # Detach parameters to avoid building computation graph twice
    with torch.no_grad():
        params = model.params_to_single_tensor().detach().to(device).requires_grad_(True)
    
    P = params.shape[0]
    
    # Initialize empty Jacobian tensor
    full_jacobian = torch.zeros((T, M, S, P), device=device)
    
    # Process in batches along the T dimension
    for batch_start in range(0, T, batch_size):
        batch_end = min(batch_start + batch_size, T)
        batch_samples = y_sample_TMS[batch_start:batch_end].detach()
        
        # Get log probs function with no_grad except for params
        def get_log_probs_baked(param_P):
            with torch.set_grad_enabled(True):
                distribution = model.build_from_single_tensor(param_P)
                return distribution.log_prob(batch_samples)
        
        # Compute Jacobian for this batch
        batch_jacobian = torch.autograd.functional.jacobian(
            get_log_probs_baked,
            params,
            strategy='forward-mode',
            vectorize=True
        )
        
        # Store in the full Jacobian tensor
        full_jacobian[batch_start:batch_end] = batch_jacobian.detach()
        
        # Delete intermediate tensors and clear cache
        del batch_jacobian
        torch.cuda.empty_cache()
        
        # Print memory usage for debugging
        if device.type == 'cuda':
            print(f"Batch {batch_start}-{batch_end}: GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
    return full_jacobian

def compute_jacobian_per_parameter(model, y_sample_TMS, device, batch_size=None):
    """Compute Jacobian per parameter to reduce memory usage"""
    with torch.no_grad():
        params = model.params_to_single_tensor().detach().to(device)
        y_sample_detached = y_sample_TMS.detach()
    
    T, M, S = y_sample_TMS.shape
    P = params.shape[0]
    
    # Initialize the output Jacobian tensor
    jacobian = torch.zeros((T, M, S, P), device=device)
    
    # Add batch processing for the per-parameter approach as well
    if batch_size is None:
        batch_size = max(1, T // 10)  # Default to 10 batches
    
    for i in range(P):
        # Log progress for large parameter counts
        if P > 100 and i % 10 == 0:
            print(f"Processing parameter {i}/{P}")
        
        # Free up memory
        torch.cuda.empty_cache()
        gc.collect()
        
        # Process this parameter in batches
        for batch_start in range(0, T, batch_size):
            batch_end = min(batch_start + batch_size, T)
            batch_samples = y_sample_detached[batch_start:batch_end]
            
            # Create a new parameter tensor for this batch
            param_i = params[i].clone().detach().requires_grad_(True)
            
            def single_param_log_prob(param_scalar):
                with torch.set_grad_enabled(True):
                    param_copy = params.clone().detach()
                    param_copy[i] = param_scalar
                    distribution = model.build_from_single_tensor(param_copy)
                    return distribution.log_prob(batch_samples)
            
            # Compute Jacobian for this parameter and batch
            jac_i_batch = torch.autograd.functional.jacobian(
                single_param_log_prob,
                param_i,
                strategy='forward-mode',
                vectorize=True
            )
            
            # Store in the full Jacobian tensor
            jacobian[batch_start:batch_end, :, :, i] = jac_i_batch.detach()
            
            # Explicitly delete to free memory
            del jac_i_batch, param_i
            torch.cuda.empty_cache()
        
        # Monitor memory usage
        if device.type == 'cuda' and i % 10 == 0:
            print(f"Parameter {i}: GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    return jacobian

def train_epoch(model, optimizer, K, threshold, train_T, M_score_func, M_action, train_y_TS, 
                perturbed_top_K_func, bpr_weight, nll_weight, device, 
                jacobian_method='accumulating', update=True, batch_size=10, debug_memory=True):
    # Print memory usage if debugging
    if debug_memory and device.type == 'cuda':
        print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    optimizer.zero_grad()
    model = model.to(device)
    
    # Create a checkpoint of model state we can restore if needed
    model_state = {k: v.clone().detach() for k, v in model.state_dict().items()}
    
    # Sample data with gradient tracking disabled
    with torch.no_grad():
        mix_model = model()
        y_sample_TMS = mix_model.sample((train_T, M_score_func)).to(device)
        y_sample_action_TMS = y_sample_TMS.clone()
        
        # Compute ratios with no_grad to avoid building computation graph
        ratio_rating_TMS = y_sample_action_TMS / y_sample_action_TMS.sum(dim=-1, keepdim=True)
        ratio_rating_TS = ratio_rating_TMS.mean(dim=1)
    
    # Now require gradients only where needed
    ratio_rating_TS = ratio_rating_TS.detach().clone().requires_grad_(True)
    
    # Free up memory before Jacobian computation
    del y_sample_action_TMS
    torch.cuda.empty_cache()
    gc.collect()
    
    if debug_memory and device.type == 'cuda':
        print(f"Before Jacobian: GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    # Try smaller batch size if specified batch size is too large
    current_batch_size = batch_size
    max_retries = 3
    retries = 0

    if jacobian_method == 'accumulating':
        print(f"Computing accumulating Jacobian with batch size {current_batch_size}")
        accumulated_score = compute_jacobian_accumulating(model, y_sample_TMS, device, batch_size_T=current_batch_size, batch_size_M=current_batch_size)


        # Directly use accumulated score instead of TMSP
        with torch.no_grad():
            score_func_estimator_TSP = accumulated_score * ratio_rating_TMS.unsqueeze(-1)

        # Free memory
        del accumulated_score, y_sample_TMS
        torch.cuda.empty_cache()
        gc.collect()
    else:
            
        while retries < max_retries:
            try:
                # Compute Jacobian using selected method
                if jacobian_method == 'together':
                    try:
                        print(f"Computing all-together Jacobian with batch size {current_batch_size}")
                        jac_TMSP = compute_jacobian_all_together_batched(model, y_sample_TMS, device, current_batch_size)
                    except RuntimeError as e:
                        print(f"Error computing all-together Jacobian: {e}")
                        raise
                        print("Falling back to per-parameter Jacobian...")
                        jac_TMSP = compute_jacobian_per_parameter(model, y_sample_TMS, device, current_batch_size)
                else:
                    print(f"Computing per-parameter Jacobian with batch size {current_batch_size}")
                    jac_TMSP = compute_jacobian_per_parameter(model, y_sample_TMS, device, current_batch_size)
                
                # If we got here, computation succeeded
                break
                
            except RuntimeError as e:
                if "CUDA out of memory" in str(e) and retries < max_retries - 1:
                    # Cut batch size in half and retry
                    torch.cuda.empty_cache()
                    gc.collect()
                    retries += 1
                    current_batch_size = max(1, current_batch_size // 2)
                    print(f"CUDA OOM error. Retrying with smaller batch size: {current_batch_size}")
                else:
                    # If we've exhausted retries or it's not an OOM error, re-raise
                    print(f"Unrecoverable error: {e}")
                    # Try to restore model parameters
                    model.load_state_dict(model_state)
                    raise
        

        
        # Score function estimator - detach Jacobian to prevent building huge graph
        with torch.no_grad():
            score_func_estimator_TMSP = jac_TMSP.detach() * ratio_rating_TMS.unsqueeze(-1)
            score_func_estimator_TSP = score_func_estimator_TMSP.mean(dim=1)
    
        # Free memory
        del jac_TMSP, score_func_estimator_TMSP, y_sample_TMS
        torch.cuda.empty_cache()
        gc.collect()

    if debug_memory and device.type == 'cuda':
        print(f"After Jacobian: GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    if debug_memory and device.type == 'cuda':
        print(f"Before BPR: GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    # Convert training data to tensor once and reuse
    train_y_tensor = torch.tensor(train_y_TS, device=device)
    
    # BPR calculation - this part requires gradients
    positive_bpr_T = torch_bpr_uncurried(
        ratio_rating_TS, 
        train_y_tensor, 
        K=K, 
        perturbed_top_K_func=perturbed_top_K_func
    )
    
    if debug_memory and device.type == 'cuda':
        print(f"After BPR: GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    # Loss calculation
    if nll_weight > 0:
        bpr_threshold_diff_T = positive_bpr_T - threshold
        negative_bpr_loss = torch.mean(-bpr_threshold_diff_T * (bpr_threshold_diff_T < 0))
    else:
        negative_bpr_loss = torch.mean(-positive_bpr_T)
    
    # Compute NLL loss with fresh model instance to avoid linking computation graphs
    with torch.set_grad_enabled(True):
        fresh_mix_model = model()
        nll = torch.mean(-fresh_mix_model.log_prob(train_y_tensor))
    
    loss = bpr_weight * negative_bpr_loss + nll_weight * nll
    
    if debug_memory and device.type == 'cuda':
        print(f"Before backward: GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    # Backpropagation with gradient clipping to prevent exploding gradients
    loss.backward()
    
    # Clip gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    
    if update:
        optimizer.step()
    
    # Free memory
    torch.cuda.empty_cache()
    gc.collect()
    
    if debug_memory and device.type == 'cuda':
        print(f"Final GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    return loss.detach(), torch.mean(positive_bpr_T).detach(), nll.detach(), mix_model

def main(T=1000, S=20, K=10, step_size=0.01, epochs=10, bpr_weight=1.0, nll_weight=1.0, seed=360,
         epsilon=0.55, num_components=4, perturbed_noise=0.05, perturbation_samples=100,
         score_func_samples=100, device='cuda', jacobian_method=True, batch_size=10,
         debug_memory=True, detect_anomaly=False):
    
    set_seed(seed)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Enable anomaly detection if requested (helps debug CUDA errors)
    if detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
        print("PyTorch anomaly detection enabled")
    
    # Print device and memory info
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Initial allocated GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    # Start with clean memory
    torch.cuda.empty_cache()
    gc.collect()

    # Create distributions for sampling
    dist_S = [Normal(loc, 0.3) for loc in [10, 20, 30, 40, 50, 60, 100]]

    # Generate training data with no_grad to prevent memory leaks
    with torch.no_grad():
        train_y_TS = np.zeros((T, S))
        for s in range(S):
            dist = dist_S[s % len(dist_S)]
            train_y_TS[:, s] = dist.sample((T,)).cpu().numpy()
    
    # Memory debugging
    if debug_memory and device.type == 'cuda':
        print(f"After data generation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Initialize model with lower precision if on GPU to save memory
    model = MixtureOfTruncNormModel(num_components=num_components, S=S, low=0, high=150)
    
    # Print model parameters
    param_count = sum(p.numel() for p in model.parameters())
    print(f'Number of parameters: {param_count}')
    
    # Suggest smaller batch size for large models
    if param_count > 10000 and batch_size > 5:
        suggested_batch = max(1, min(batch_size, 5))
        print(f"Model has {param_count} parameters. Consider using a smaller batch size (e.g., --batch_size {suggested_batch})")
    
    # Move model to device
    model = model.to(device)
    
    # Initialize optimizer with gradient clipping
    optimizer = torch.optim.Adam(model.parameters(), lr=step_size)
    
    # Optional: Use a learning rate scheduler for stability
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )

    # Create top-k function
    top_k_func = partial(top_k_onehot_indicator, k=K)
    
    # Debug the perturbation samples count if it's very high
    if perturbation_samples > 100:
        print(f"Warning: High perturbation_samples ({perturbation_samples}) may cause memory issues")
    
    # Create perturbed top-K function
    perturbed_top_K_func = perturbed(top_k_func, sigma=perturbed_noise, num_samples=perturbation_samples, device=device)
    
    # Print training configuration
    print(f"Training config: T={T}, S={S}, K={K}, samples={score_func_samples}, batch_size={batch_size}")
    print(f"Using {jacobian_method} Jacobian computation")

    # Try to catch and recover from memory errors during training
    epoch_batch_sizes = []
    
    for epoch in range(epochs):
        print(f'EPOCH: {epoch}')
        
        # Clear memory before each epoch
        torch.cuda.empty_cache()
        gc.collect()
        
        # If we've had to adjust batch size in previous epochs, use the smallest successful one
        effective_batch_size = min(batch_size, *epoch_batch_sizes) if epoch_batch_sizes else batch_size
        
        if effective_batch_size != batch_size:
            print(f"Using reduced batch size of {effective_batch_size} based on previous epochs")
        
        try:
            # Run one epoch of training
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
                jacobian_method=jacobian_method,
                batch_size=effective_batch_size,
                debug_memory=debug_memory
            )
            
            # If we succeeded, remember this batch size
            epoch_batch_sizes.append(effective_batch_size)
            
            # Print results
            print(f"Loss: {loss:.4f}, BPR: {bpr:.4f}, NLL: {nll:.4f}")
            
            # Update learning rate scheduler
            scheduler.step(loss)
            
        except RuntimeError as e:
            # If we hit an unrecoverable error during training
            print(f"Error during epoch {epoch}: {e}")
            raise
            
            if "CUDA out of memory" in str(e) and effective_batch_size > 1:
                # Try an even smaller batch size next epoch
                new_batch_size = max(1, effective_batch_size // 2)
                print(f"Reducing batch size to {new_batch_size} for next epoch")
                epoch_batch_sizes.append(new_batch_size)
                continue
            else:
                # For non-OOM errors or if batch size is already 1, we can't recover
                print("Unrecoverable error, stopping training")
                break
        
        # Print memory status after each epoch
        if device.type == 'cuda':
            print(f"End of epoch {epoch}: GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default='cuda', help='Device to run on (cuda or cpu)')
    
    parser.add_argument("--batch_size", type=int, default=10, help='Batch size for Jacobian computation')
    parser.add_argument("--T", type=int, default=1000, help='Number of training samples')
    parser.add_argument("--S", type=int, default=20, help='Number of score function components')
    parser.add_argument("--K", type=int, default=10, help='Top-K value for BPR calculation')
    parser.add_argument("--score_func_samples", type=int, default=100, help='Number of score function samples')
    parser.add_argument("--debug_memory", action='store_true', help='Print memory usage information')
    parser.add_argument("--detect_anomaly", action='store_true', help='Enable PyTorch anomaly detection')
    parser.add_argument("--seed", type=int, default=360, help='Random seed')
    parser.add_argument("--jacobian_method", type=str, choices=['together', 'per_parameter', 'accumulating'], default='together', help='Method to compute the Jacobian: together, per_parameter, or accumulating')
    parser.add_argument("--perturbation_samples", type=int, default=100, help='Number of perturbation samples for BPR calculation')
    parser.add_argument("--num_components", type=int, default=4, help='Number of components in the mixture model')
    args = parser.parse_args()

    print("Running with optimized memory settings:")
    main(
        epochs=args.epochs, 
        device=args.device, 
        jacobian_method=args.jacobian_method,
        batch_size=args.batch_size,
        T=args.T,
        S=args.S,
        K=args.K,
        score_func_samples=args.score_func_samples,
        debug_memory=args.debug_memory,
        detect_anomaly=args.detect_anomaly,
        seed=args.seed,
        num_components=args.num_components
    )