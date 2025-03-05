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

def compute_gradient_chunks(model, y_sample_TMS, device, t_batch_size=5, m_batch_size=5, s_batch_size=None):
    """
    Compute gradients directly without storing Jacobian tensors.
    This method:
    1. Processes data in small chunks (t_batch × m_batch × s_batch)
    2. For each chunk, computes log probs and directly accumulates gradients
    3. Avoids storing any large intermediate tensors (no T×S×P or T×M×S×P)
    
    Returns: None - gradients are directly accumulated in model parameters
    """
    T, M, S = y_sample_TMS.shape
    
    # Use all S if s_batch_size not specified
    if s_batch_size is None:
        s_batch_size = S
    
    # Detach parameters and zero gradients before we start
    model.zero_grad()
    
    # Process in small t_batch × m_batch × s_batch chunks
    for t_start in range(0, T, t_batch_size):
        t_end = min(t_start + t_batch_size, T)
        t_size = t_end - t_start
        
        for m_start in range(0, M, m_batch_size):
            m_end = min(m_start + m_batch_size, M)
            m_size = m_end - m_start
            
            for s_start in range(0, S, s_batch_size):
                s_end = min(s_start + s_batch_size, S)
                s_size = s_end - s_start
                
                # Get this small chunk of samples
                # Shape: [t_size, m_size, s_size]
                chunk_samples = y_sample_TMS[t_start:t_end, m_start:m_end, s_start:s_end].detach()
                
                # Compute distribution for this chunk
                distribution = model()
                
                # Compute log probabilities for this chunk
                log_probs = distribution.log_prob(chunk_samples)
                
                # Compute mean log probability across this chunk
                # Scale by total size to get proper contribution to overall gradient
                mean_log_prob = log_probs.mean() * (t_size * m_size * s_size) / (T * M * S)
                
                # Backpropagate to accumulate gradients
                mean_log_prob.backward()
                
                # Clear memory
                del chunk_samples, distribution, log_probs, mean_log_prob
                torch.cuda.empty_cache()
            
            # Print progress and memory usage
            if device.type == 'cuda':
                print(f"Processed chunk T[{t_start}:{t_end}], M[{m_start}:{m_end}] - "
                      f"Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    # No return value - gradients are accumulated in model parameters

def compute_loss_and_backward_chunked(model, optimizer, K, threshold, train_T, M_score_func, 
                                      train_y_TS, perturbed_top_K_func, bpr_weight, nll_weight, 
                                      device, t_batch_size=5, m_batch_size=5, s_batch_size=None,
                                      debug_memory=True, update=True):
    """
    Compute loss and gradients in chunks without ever storing large tensors.
    This completely avoids creating any Jacobian tensors.
    """
    # Start fresh
    optimizer.zero_grad()
    model = model.to(device)
    S = train_y_TS.shape[1]
    
    # Use all S if s_batch_size not specified
    if s_batch_size is None:
        s_batch_size = S
    
    # Memory debugging
    if debug_memory and device.type == 'cuda':
        print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    # Sample data in batches to avoid large tensors
    max_m_batch = min(m_batch_size, M_score_func)
    total_samples = torch.zeros((train_T, S), device=device)
    
    # Keep track of all sample means for BPR
    with torch.no_grad():
        for m_start in range(0, M_score_func, max_m_batch):
            m_end = min(m_start + max_m_batch, M_score_func)
            m_size = m_end - m_start
            
            # Generate this batch of samples
            mix_model = model()
            y_batch = mix_model.sample((train_T, m_size)).to(device)
            
            # Normalize samples (compute ratio)
            y_sum = y_batch.sum(dim=-1, keepdim=True)
            ratio_batch = y_batch / y_sum
            
            # Accumulate mean (will divide by M_score_func at the end)
            total_samples += ratio_batch.sum(dim=1) * (m_size / M_score_func)
            
            # Clear memory
            del mix_model, y_batch, y_sum, ratio_batch
            torch.cuda.empty_cache()
            
            # Print progress
            if debug_memory and device.type == 'cuda':
                print(f"Generated samples batch {m_start}/{M_score_func} - "
                      f"Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    # Final sample ratios with gradients enabled
    ratio_rating_TS = total_samples.detach().clone().requires_grad_(True)
    
    # Clear memory before loss computation
    del total_samples
    torch.cuda.empty_cache()
    gc.collect()
    
    # Compute BPR loss in t-batches to save memory
    train_y_tensor = torch.tensor(train_y_TS, device=device)
    
    # Initialize loss components
    bpr_loss_sum = 0
    bpr_count = 0
    
    for t_start in range(0, train_T, t_batch_size):
        t_end = min(t_start + t_batch_size, train_T)
        t_size = t_end - t_start
        
        # Compute BPR for this batch
        positive_bpr_batch = torch_bpr_uncurried(
            ratio_rating_TS[t_start:t_end], 
            train_y_tensor[t_start:t_end], 
            K=K, 
            perturbed_top_K_func=perturbed_top_K_func
        )
        
        # Calculate BPR loss for this batch
        if nll_weight > 0:
            bpr_threshold_diff = positive_bpr_batch - threshold
            negative_bpr_loss_batch = torch.sum(-bpr_threshold_diff * (bpr_threshold_diff < 0))
        else:
            negative_bpr_loss_batch = torch.sum(-positive_bpr_batch)
        
        # Add to running total (will divide by total count at the end)
        bpr_loss_sum += negative_bpr_loss_batch
        bpr_count += t_size
        
        # Track average BPR for reporting
        if t_start == 0:
            avg_bpr = torch.mean(positive_bpr_batch).detach()
            
        # Clear memory
        del positive_bpr_batch
        
        if debug_memory and device.type == 'cuda' and t_start % 100 == 0:
            print(f"Computed BPR for batch {t_start}/{train_T} - "
                  f"Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    # Finalize BPR loss
    negative_bpr_loss = bpr_loss_sum / bpr_count
    
    # Compute NLL loss in chunks to save memory
    nll_loss_sum = 0
    nll_count = 0
    
    for t_start in range(0, train_T, t_batch_size):
        t_end = min(t_start + t_batch_size, train_T)
        t_size = t_end - t_start
        
        for s_start in range(0, S, s_batch_size):
            s_end = min(s_start + s_batch_size, S)
            s_size = s_end - s_start
            
            # Get distribution
            mix_model = model()
            
            # Compute NLL for this chunk
            chunk_nll = -mix_model.log_prob(train_y_tensor[t_start:t_end, s_start:s_end])
            nll_loss_sum += torch.sum(chunk_nll)
            nll_count += t_size * s_size
            
            # Clear memory
            del mix_model, chunk_nll
            torch.cuda.empty_cache()
            
        if debug_memory and device.type == 'cuda' and t_start % 100 == 0:
            print(f"Computed NLL for batch {t_start}/{train_T} - "
                  f"Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    # Finalize NLL loss
    nll_loss = nll_loss_sum / nll_count
    nll = nll_loss.detach()  # Save for reporting
    
    # Combine losses and backpropagate
    if bpr_weight > 0:
        bpr_grad = torch.autograd.grad(
            negative_bpr_loss, ratio_rating_TS, retain_graph=False, create_graph=False
        )[0]
    else:
        bpr_grad = torch.zeros_like(ratio_rating_TS)
    
    if nll_weight > 0:
        # Directly compute final NLL gradient without retain_graph
        nll_loss.backward()
    
    # If we need BPR gradients, compute them
    if bpr_weight > 0:
        # Now compute actual model gradients from ratio_rating_TS gradients
        # Use score function estimator in chunks
        
        # Reset optimizer for clean accumulation
        optimizer.zero_grad()
        
        # Process in small chunks
        for t_start in range(0, train_T, t_batch_size):
            t_end = min(t_start + t_batch_size, train_T)
            
            for m_start in range(0, max(1, M_score_func), max(1, m_batch_size)):
                m_end = min(m_start + m_batch_size, max(1, M_score_func))
                m_size = max(1, m_end - m_start)
                
                # Generate this batch of samples with gradients
                mix_model = model()
                y_batch = mix_model.sample((t_end - t_start, m_size)).to(device)
                
                # Normalize samples (compute ratio)
                y_sum = y_batch.sum(dim=-1, keepdim=True)
                ratio_batch = y_batch / y_sum
                
                # Compute log probs
                log_probs = mix_model.log_prob(y_batch)
                
                # Weight log probs by BPR gradients for this batch
                # This is the score function estimator
                weighted_log_probs = (
                    log_probs * (ratio_batch * bpr_grad[t_start:t_end].unsqueeze(1)).sum(dim=-1)
                ).mean() * bpr_weight
                
                # Backpropagate
                weighted_log_probs.backward()
                
                # Clear memory
                del mix_model, y_batch, y_sum, ratio_batch, log_probs, weighted_log_probs
                torch.cuda.empty_cache()
    
    # Memory cleanup
    del ratio_rating_TS, bpr_loss_sum, negative_bpr_loss
    if bpr_weight > 0:
        del bpr_grad
    torch.cuda.empty_cache()
    
    # Clip gradients to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    
    # Update model if requested
    if update:
        optimizer.step()
    
    # Final memory cleanup
    torch.cuda.empty_cache()
    gc.collect()
    
    # Calculate loss for reporting
    loss = bpr_weight * negative_bpr_loss.detach() + nll_weight * nll
    
    return loss.detach(), avg_bpr, nll, (t_batch_size, m_batch_size, s_batch_size)

def train_epoch_ultra_memory_efficient(model, optimizer, K, threshold, train_T, M_score_func,
                                      train_y_TS, perturbed_top_K_func, bpr_weight, nll_weight,
                                      device, t_batch_size=5, m_batch_size=5, s_batch_size=None,
                                      debug_memory=True, update=True):
    """
    Ultra memory-efficient training that avoids creating large tensors completely.
    Uses a direct gradient computation approach rather than creating Jacobians.
    """
    # Print initial memory usage
    if debug_memory and device.type == 'cuda':
        print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    # Try smaller batch sizes if needed
    current_t_batch_size = t_batch_size
    current_m_batch_size = m_batch_size
    current_s_batch_size = s_batch_size
    max_retries = 3
    retries = 0
    
    # Try to compute with current batch sizes, reduce if OOM occurs
    while retries < max_retries:
        try:
            # Compute loss and gradients directly in chunks
            return compute_loss_and_backward_chunked(
                model, optimizer, K, threshold, train_T, M_score_func,
                train_y_TS, perturbed_top_K_func, bpr_weight, nll_weight,
                device, current_t_batch_size, current_m_batch_size, current_s_batch_size,
                debug_memory, update
            )
        
        except RuntimeError as e:
            raise
            if "CUDA out of memory" in str(e) and retries < max_retries - 1:
                # Clear memory
                torch.cuda.empty_cache()
                gc.collect()
                
                # Reduce batch sizes
                retries += 1
                
                # Try reducing the largest dimension first
                sizes = [current_t_batch_size, current_m_batch_size]
                if current_s_batch_size is not None:
                    sizes.append(current_s_batch_size)
                
                max_idx = sizes.index(max(sizes))
                
                if max_idx == 0 and current_t_batch_size > 1:
                    current_t_batch_size = max(1, current_t_batch_size // 2)
                    print(f"Reducing t_batch_size to {current_t_batch_size}")
                elif max_idx == 1 and current_m_batch_size > 1:
                    current_m_batch_size = max(1, current_m_batch_size // 2)
                    print(f"Reducing m_batch_size to {current_m_batch_size}")
                elif current_s_batch_size is not None and current_s_batch_size > 1:
                    current_s_batch_size = max(1, current_s_batch_size // 2)
                    print(f"Reducing s_batch_size to {current_s_batch_size}")
                else:
                    # If all batch sizes are 1, we can't reduce further
                    print("All batch sizes at minimum. Can't reduce further.")
                    raise
                
                print(f"CUDA OOM error. Retrying with t_batch={current_t_batch_size}, "
                      f"m_batch={current_m_batch_size}, s_batch={current_s_batch_size}")
            else:
                # Re-raise if not OOM or out of retries
                print(f"Unrecoverable error: {e}")
                raise
    
    # This should not be reached if all retries fail, but provide a fallback
    raise RuntimeError("Failed to compute with minimum batch sizes")

def main(T=1000, S=20, K=10, step_size=0.01, epochs=10, bpr_weight=1.0, nll_weight=1.0,
         seed=360, epsilon=0.55, num_components=4, perturbed_noise=0.05,
         perturbation_samples=100, score_func_samples=100, device='cuda',
         t_batch_size=5, m_batch_size=5, s_batch_size=None,
         debug_memory=True, detect_anomaly=False):
    
    set_seed(seed)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Enable anomaly detection if requested
    if detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
        print("PyTorch anomaly detection enabled")
    
    # Print device and memory info
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    # Start with clean memory
    torch.cuda.empty_cache()
    gc.collect()

    # Create distributions for sampling
    dist_S = [Normal(loc, 0.3) for loc in [10, 20, 30, 40, 50, 60, 100]]

    # Generate training data with no_grad
    with torch.no_grad():
        train_y_TS = np.zeros((T, S))
        for s in range(S):
            dist = dist_S[s % len(dist_S)]
            train_y_TS[:, s] = dist.sample((T,)).cpu().numpy()
    
    # Memory debugging
    if debug_memory and device.type == 'cuda':
        print(f"After data generation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Initialize model
    model = MixtureOfTruncNormModel(num_components=num_components, S=S, low=0, high=150)
    
    # Print model parameters
    param_count = sum(p.numel() for p in model.parameters())
    print(f'Number of parameters: {param_count}')
    
    # Suggest smaller batch sizes for large models
    if param_count > 10000:
        suggested_t_batch = max(1, min(t_batch_size, 5))
        suggested_m_batch = max(1, min(m_batch_size, 5))
        print(f"Model has {param_count} parameters. Consider using smaller batch sizes "
              f"(e.g., --t_batch_size {suggested_t_batch} --m_batch_size {suggested_m_batch})")
    
    # Move model to device
    model = model.to(device)
    
    # Initialize optimizer with gradient clipping
    optimizer = torch.optim.Adam(model.parameters(), lr=step_size)
    
    # Learning rate scheduler for stability
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )

    # Create top-k function
    top_k_func = partial(top_k_onehot_indicator, k=K)
    
    # Debug the perturbation samples count
    if perturbation_samples > 100:
        print(f"Warning: High perturbation_samples ({perturbation_samples}) may cause memory issues")
    
    # Create perturbed top-K function
    perturbed_top_K_func = perturbed(top_k_func, sigma=perturbed_noise, num_samples=perturbation_samples, device=device)
    
    # Print training configuration
    print(f"Training config: T={T}, S={S}, K={K}, samples={score_func_samples}")
    print(f"Initial batch sizes: t_batch={t_batch_size}, m_batch={m_batch_size}, s_batch={s_batch_size}")

    # Track successful batch sizes
    successful_t_batch_sizes = []
    successful_m_batch_sizes = []
    successful_s_batch_sizes = []
    
    for epoch in range(epochs):
        print(f'EPOCH: {epoch}')
        
        # Clear memory before each epoch
        torch.cuda.empty_cache()
        gc.collect()
        
        # Use the smallest successful batch sizes from previous epochs if available
        effective_t_batch_size = min([t_batch_size] + successful_t_batch_sizes) if successful_t_batch_sizes else t_batch_size
        effective_m_batch_size = min([m_batch_size] + successful_m_batch_sizes) if successful_m_batch_sizes else m_batch_size
        effective_s_batch_size = min([s_batch_size] + successful_s_batch_sizes) if s_batch_size is not None and successful_s_batch_sizes else s_batch_size
        
        if (effective_t_batch_size != t_batch_size or 
            effective_m_batch_size != m_batch_size or 
            (s_batch_size is not None and effective_s_batch_size != s_batch_size)):
            print(f"Using reduced batch sizes: t_batch={effective_t_batch_size}, "
                  f"m_batch={effective_m_batch_size}, s_batch={effective_s_batch_size}")
        
        try:
            # Run one epoch of training with ultra memory-efficient approach
            loss, bpr, nll, (actual_t_batch, actual_m_batch, actual_s_batch) = train_epoch_ultra_memory_efficient(
                model,
                optimizer,
                K,
                epsilon,
                T,
                score_func_samples,
                train_y_TS,
                perturbed_top_K_func,
                bpr_weight,
                nll_weight,
                device=device,
                t_batch_size=effective_t_batch_size,
                m_batch_size=effective_m_batch_size,
                s_batch_size=effective_s_batch_size,
                debug_memory=debug_memory,
                update=True
            )
            
            # Remember successful batch sizes
            successful_t_batch_sizes.append(actual_t_batch)
            successful_m_batch_sizes.append(actual_m_batch)
            if actual_s_batch is not None:
                successful_s_batch_sizes.append(actual_s_batch)
            
            # Print results
            print(f"Loss: {loss:.4f}, BPR: {bpr:.4f}, NLL: {nll:.4f}")
            print(f"Successful batch sizes: t_batch={actual_t_batch}, "
                  f"m_batch={actual_m_batch}, s_batch={actual_s_batch}")
            
            # Update learning rate scheduler
            scheduler.step(loss)
            
        except RuntimeError as e:
            print(f"Error during epoch {epoch}: {e}")
            raise
            # Continue training with reduced batch sizes next epoch if possible
            minimum_reached = (
                (effective_t_batch_size <= 1) and 
                (effective_m_batch_size <= 1) and 
                (effective_s_batch_size is None or effective_s_batch_size <= 1)
            )
            
            if "CUDA out of memory" in str(e) and not minimum_reached:
                # Reduce batch sizes for next epoch
                if effective_t_batch_size > 1:
                    successful_t_batch_sizes.append(max(1, effective_t_batch_size // 2))
                if effective_m_batch_size > 1:
                    successful_m_batch_sizes.append(max(1, effective_m_batch_size // 2))
                if effective_s_batch_size is not None and effective_s_batch_size > 1:
                    successful_s_batch_sizes.append(max(1, effective_s_batch_size // 2))
                
                print("Reduced batch sizes for next epoch")
                continue
            else:
                # For non-OOM errors or if batch sizes are already at minimum
                print("Unrecoverable error, stopping training")
                break
        
        # Print memory status after each epoch
        if device.type == 'cuda':
            print(f"End of epoch {epoch}: GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default='cuda', help='Device to run on (cuda or cpu)')
    parser.add_argument("--t_batch_size", type=int, default=5, help='Batch size for T dimension')
    parser.add_argument("--m_batch_size", type=int, default=5, help='Batch size for M dimension')
    parser.add_argument("--s_batch_size", type=int, default=None, help='Batch size for S dimension')
    parser.add_argument("--T", type=int, default=1000, help='Number of training samples')
    parser.add_argument("--S", type=int, default=20, help='Number of score function components')
    parser.add_argument("--K", type=int, default=10, help='Top-K value for BPR calculation')
    parser.add_argument("--score_func_samples", type=int, default=100, help='Number of score function samples')
    parser.add_argument("--debug_memory", action='store_true', help='Print memory usage information')
    parser.add_argument("--detect_anomaly", action='store_true', help='Enable PyTorch anomaly detection')
    parser.add_argument("--seed", type=int, default=360, help='Random seed')
    parser.add_argument("--bpr_weight", type=float, default=1.0, help='Weight for BPR loss')
    parser.add_argument("--nll_weight", type=float, default=1.0, help='Weight for NLL loss')
    
    args = parser.parse_args()

    print("Running with ultra memory-efficient gradient computation:")
    main(
        epochs=args.epochs, 
        device=args.device, 
        t_batch_size=args.t_batch_size,
        m_batch_size=args.m_batch_size,
        s_batch_size=args.s_batch_size,
        T=args.T,
        S=args.S,
        K=args.K,
        score_func_samples=args.score_func_samples,
        debug_memory=args.debug_memory,
        detect_anomaly=args.detect_anomaly,
        seed=args.seed,
        bpr_weight=args.bpr_weight,
        nll_weight=args.nll_weight
    )