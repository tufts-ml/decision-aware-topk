import torch

import torch.nn as nn
from torch.distributions import Categorical, Poisson, MixtureSameFamily, NegativeBinomial
from torch_distributions import TruncatedNormal


class RatioRatingFunction(torch.autograd.Function):
    """
    A custom Function that:
      - Takes your model parameters (rolled up) and any other inputs
      - Samples from the distribution
      - Computes ratio_rating
      - On backward, uses forward-mode jacobian * ratio_rating for the gradient wrt params
    """

    @staticmethod
    def forward(ctx,
                params_single_tensor,      # (param_dim,) single rolled-up param vector
                feat,                      # input features for this time step or batch
                time,
                M_score_func,
                model_build_from_params):
        """
        forward: returns ratio_rating_TS as a normal tensor that autograd can track.
        """

        with torch.no_grad():
            # 1) Rebuild the distribution from the param vector
            dist = model_build_from_params(params_single_tensor, feat, time)

            # 2) Sample from the distribution
            #    Suppose we get shape [M_score_func, B, D], for example
            y_sample_MBD = dist.sample((M_score_func,))
            # reorder dims so it's [B, M_score_func, D] if you like
            y_sample_BMD = y_sample_MBD.permute(1,0,2)  

            # 3) ratio_rating_BMD = y_sample / (sum(...) + 1)
            denom_BM1 = y_sample_BMD.sum(dim=-1, keepdim=True) + 1.0
            ratio_rating_BMD = y_sample_BMD / denom_BM1

            # We'll just average over M to get ratio_rating_BD
            ratio_rating_BD = ratio_rating_BMD.mean(dim=1)

        # 4) We must save everything needed for backward
        #    - the original param vector
        #    - the sample we used
        #    - the ratio_rating_BMD to help us replicate your jacobian logic
        #    - the function needed to rebuild distributions
        ctx.save_for_backward(params_single_tensor, y_sample_BMD, ratio_rating_BMD)
        ctx.model_build_from_params = model_build_from_params
        ctx.feat = feat
        ctx.time = time
        ctx.M_score_func = M_score_func

        # Return ratio_rating_BD as a normal differentiable tensor
        # We'll let PyTorch see it as requires_grad, so chain rule works.
        ratio_rating_BD.requires_grad_(True)
        return ratio_rating_BD

    @staticmethod
    def backward(ctx, grad_output):
        """
        backward: 
         - grad_output is d(Loss)/d(ratio_rating_BD), shape [B, D]
         - We want d(Loss)/d(params_single_tensor).
         - We'll do the forward-mode jac * ratio logic from your snippet.
         - Then we multiply by grad_output for chain rule.
         - We return that as the gradient w.r.t. 'params_single_tensor'.
        """

        # 1) Recover saved items
        params_single_tensor, y_sample_BMD, ratio_rating_BMD = ctx.saved_tensors
        model_build_from_params = ctx.model_build_from_params
        feat = ctx.feat
        time = ctx.time
        M_score_func = ctx.M_score_func

        # 2) We re-run the log_prob under the same distribution, but *as a function of param_tensor*
        def get_log_probs_baked(param_tensor):
            """
            Return log_prob of y_sample_BMD, shape [B, M, D],
            so the result is also [B, M, D].
            """
            distribution = model_build_from_params(param_tensor, feat, time)
            # reorder for log_prob => shape: [M, B, D]
            #print(f'BMD: {y_sample_BMD.size()}')
            y_sample_MBD = y_sample_BMD.permute(1,0,2)  
            log_probs_MBD = distribution.log_prob(y_sample_MBD)
            # permute back => shape: [B, M, D]
            return log_probs_MBD.permute(1,0,2)

        # 3) Compute Jacobian wrt the parameters using forward-mode
        #    shape of jac => [B, M, D, param_dim] if get_log_probs_baked
        #    returns shape [B, M, D].
        jac_BMDP = torch.autograd.functional.jacobian(
            get_log_probs_baked,
            (params_single_tensor,),
            strategy='forward-mode',
            vectorize=True
        )
        # jac_BMDP is a tuple containing the actual tensor if we pass a 1-tuple
        # Typically you'd do: jac_BMDP = jac_BMDP[0] if that is how your code unpacks.

        if isinstance(jac_BMDP, (tuple, list)):
            jac_BMDP = jac_BMDP[0]  # unwrap from tuple

        # 4) Multiply by ratio term => "score function"
        #    ratio_rating_BMD shape [B, M, D], so we broadcast it to [B, M, D, 1]
        sf_BMDP = jac_BMDP * ratio_rating_BMD.unsqueeze(-1)

        # 5) Average over M dimension if that is your desired approach
        #    => shape [B, D, param_dim]
        sf_BDP = sf_BMDP.mean(dim=1)

        # 6) The upstream grad is shape [B, D] => we broadcast to [B, D, 1]
        grad_output_BD1 = grad_output.unsqueeze(-1)

        # 7) chain rule: d(L)/d(param) = sum_{b,d} [ sf_BDP * grad_output_BD1 ]
        #    => shape [param_dim]
        #    You can reduce over B & D by summation
        grad_params = torch.sum(sf_BDP * grad_output_BD1, dim=[0,1])

        # 8) Return gradient w.r.t. each input of `forward`
        # forward signature was (params_single_tensor, feat, time, M_score_func, build_func)
        # so we have to return a gradient for each. We only want to produce non-None
        # for the param tensor. The others we treat as constants => return None.
        d_params_single_tensor = grad_params
        return d_params_single_tensor, None, None, None, None

class NegativeBinomialRegressionModel(nn.Module):
    def __init__(self, num_locations, num_fixed_effects, low=0, high=float('inf'), device='cpu'):
        super(NegativeBinomialRegressionModel, self).__init__()
        self.num_locations = num_locations
        self.num_fixed_effects = num_fixed_effects
        self.low = low
        self.high = high
        
        # Fixed effects
        self.beta_0 = nn.Parameter(torch.randn(1))
        self.beta = nn.Parameter(torch.randn(num_fixed_effects))

        # Random effects - why is this not param?
        self.b_0 = nn.Parameter(torch.zeros(num_locations).to(device))
        self.b_1 = nn.Parameter(torch.zeros(num_locations).to(device))

        # Covariance matrix parameters
        self.log_sigma_0 = nn.Parameter(torch.randn(1))
        self.log_sigma_1 = nn.Parameter(torch.randn(1))
        self.rho = nn.Parameter(torch.randn(1))

        # probability param
        self.siginv_theta = nn.Parameter(torch.randn(1))

    def forward(self, X, time):
        # Ensure X and time have the correct shape
        assert X.shape[1] == self.num_locations, "X should have shape (num_time_points, num_locations, num_fixed_effects)"
        assert X.shape[2] == self.num_fixed_effects, "X should have shape (num_time_points, num_locations, num_fixed_effects)"
        assert time.shape == X.shape[:2], "time should have shape (num_time_points, num_locations)"

        # Calculate mu
        fixed_effects = self.beta_0 + torch.einsum('tli,i->tl', X, self.beta)
        random_intercepts = self.b_0.expand(X.shape[0], -1)
        random_slopes = self.b_1.expand(X.shape[0], -1)
        
        log_mu = fixed_effects + random_intercepts + random_slopes * time

        # Use softplus to ensure mu is positive and grows more slowly
        mu = nn.functional.softplus(log_mu)

        # Calculate theta probability
        theta = torch.nn.functional.sigmoid(self.siginv_theta)

        # Calculate logits instead of probabilities
        logits = torch.log(mu) - torch.log(theta)

        # Create and return the NegativeBinomial distribution using logits
        return NegativeBinomial(total_count=mu, probs=theta)
    def get_covariance_matrix(self):
        sigma_0 = torch.exp(self.log_sigma_0)
        sigma_1 = torch.exp(self.log_sigma_1)
        rho = torch.tanh(self.rho)  # Ensure -1 < rho < 1
        
        # Construct the covariance matrix using operations that maintain the computational graph
        cov_00 = sigma_0 * sigma_0
        cov_01 = rho * sigma_0 * sigma_1
        cov_11 = sigma_1 * sigma_1
        
        cov_matrix = torch.stack([
            torch.stack([cov_00, cov_01]),
            torch.stack([cov_01, cov_11])
        ])

        cov_matrix = torch.squeeze(cov_matrix)
        
        return cov_matrix

    def log_likelihood(self, y, X, time):
        dist = self.forward(X, time)
        log_prob = dist.log_prob(y)
        
        # Add penalty for random effects
        cov_matrix = self.get_covariance_matrix()
        random_effects = torch.stack([self.b_0, self.b_1], dim=1)
        penalty = -0.5 * torch.mean(
            torch.matmul(random_effects, torch.inverse(cov_matrix)) * random_effects
        )
        
        return torch.mean(log_prob) + penalty

    def sample(self, X, time, num_samples=1):
        dist = self.forward(X, time)
        samples = dist.sample((num_samples,))
        return torch.clamp(samples, self.low, self.high)

    def params_to_single_tensor(self):
        return torch.cat([param.view(-1) for param in self.parameters()])

    def single_tensor_to_params(self, single_tensor):
        params = []
        idx = 0
        for param in self.parameters():
            num_elements = param.numel()
            param_data = single_tensor[idx:idx+num_elements].view(param.shape)
            params.append(param_data)
            idx += num_elements
        return params

    def update_params(self, single_tensor):
        params = self.single_tensor_to_params(single_tensor)
        for param, new_data in zip(self.parameters(), params):
            param.data = new_data

    def build_from_single_tensor(self, single_tensor, X, time):
        beta_0, beta, b_0, b_1, log_sigma_0, log_sigma_1, rho, siginv_theta = self.single_tensor_to_params(single_tensor)
        fixed_effects = beta_0 + torch.einsum('tli,i->tl', X, beta)
        random_intercepts = b_0.expand(X.shape[0], -1)
        random_slopes = b_1.expand(X.shape[0], -1)
        
        log_mu = fixed_effects + random_intercepts + random_slopes * time

        # Use softplus to ensure mu is positive and grows more slowly
        mu = nn.functional.softplus(log_mu)

        # Calculate theta probability
        theta = torch.nn.functional.sigmoid(siginv_theta)

        
        return NegativeBinomial(total_count=mu, probs=theta)

    def plot_learned(self, X, time, data=None, ax=None):
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
        
        dist = self.forward(X, time)
        samples = dist.sample((1000,)).view(-1).detach().numpy()
        
        if ax is None:
            fig, ax = plt.subplots()
        
        sns.kdeplot(samples, bw_adjust=0.5, ax=ax)
        
        if data is not None:
            sns.histplot(data.flatten(), stat='density', bins=50, color='red', alpha=0.5, ax=ax)
        
        plt.show()
        
        return ax
    
class MixtureOfPoissonsModel(nn.Module):
    def __init__(self, num_components=4, S=12):
        super(MixtureOfPoissonsModel, self).__init__()
        self.num_components = num_components
        self.S = S
        
        # Initialize the log rates and mixture probabilities as learnable parameters
        self.log_poisson_rates = nn.Parameter(torch.rand(num_components), requires_grad=True)  # Initialize log rates
        self.mixture_probs = nn.Parameter(torch.rand(S, num_components), requires_grad=True)  # Initialize probabilities

    def params_to_single_tensor(self):
        return torch.cat([self.log_poisson_rates, self.mixture_probs.view(-1)])
    
    def single_tensor_to_params(self, single_tensor):
        log_poisson_rates = single_tensor[:self.num_components]
        mixture_probs = single_tensor[self.num_components:].view(self.S, self.num_components)
        return log_poisson_rates, mixture_probs
    
    def update_params(self, single_tensor):
        log_poisson_rates, mixture_probs = self.single_tensor_to_params(single_tensor)
        self.log_poisson_rates.data = log_poisson_rates
        self.mixture_probs.data = mixture_probs
        return
    
    def build_from_single_tensor(self, single_tensor):
        log_poisson_rates, mixture_probs = self.single_tensor_to_params(single_tensor)
        poisson_rates = torch.exp(log_poisson_rates)
        mixture_probs_normalized = torch.nn.functional.softmax(mixture_probs, dim=1)
        categorical_dist = Categorical(mixture_probs_normalized)
        expanded_rates = poisson_rates.expand(self.S, self.num_components)
        poisson_dist = Poisson(expanded_rates, validate_args=False)
        mixture_dist = MixtureSameFamily(categorical_dist, poisson_dist)
        return mixture_dist
        
    def forward(self):
        # Transform log rates to rates
        poisson_rates = torch.exp(self.log_poisson_rates)
        
        # Normalize mixture_probs to sum to 1 across the components
        mixture_probs_normalized = torch.nn.functional.softmax(self.mixture_probs, dim=1)
        
        # Create the Categorical distribution with the normalized probabilities
        categorical_dist = Categorical(mixture_probs_normalized)
        
        # Expand the Poisson rates to match the number of samples
        expanded_rates = poisson_rates.expand(self.S, self.num_components)
        
        # Create the Poisson distribution with the expanded rates
        poisson_dist = Poisson(expanded_rates, validate_args=False)
        
        # Create the MixtureSameFamily distribution
        mixture_dist = MixtureSameFamily(categorical_dist, poisson_dist,  validate_args=False)
        
        return mixture_dist

class PyEPONegativeBinomialRegressionModel(NegativeBinomialRegressionModel):
    def __init__(self, num_locations, num_fixed_effects, low=0, high=float('inf'), device='cpu', score_func_samples=100):
        super(PyEPONegativeBinomialRegressionModel, self).__init__(num_locations, num_fixed_effects, low, high, device)
        self.num_locations = num_locations
        self.num_fixed_effects = num_fixed_effects
        self.low = low
        self.high = high
        self.score_func_samples = score_func_samples
        
        # Fixed effects
        self.beta_0 = nn.Parameter(torch.randn(1))
        self.beta = nn.Parameter(torch.randn(num_fixed_effects))

        # Random effects - why is this not param?
        self.b_0 = nn.Parameter(torch.zeros(num_locations).to(device))
        self.b_1 = nn.Parameter(torch.zeros(num_locations).to(device))

        # Covariance matrix parameters
        self.log_sigma_0 = nn.Parameter(torch.randn(1))
        self.log_sigma_1 = nn.Parameter(torch.randn(1))
        self.rho = nn.Parameter(torch.randn(1))

        # probability param
        self.siginv_theta = nn.Parameter(torch.randn(1))

    def forward(self, X):
        return self.ratio_rating(X, self.score_func_samples)
    
    def log_likelihood(self, y, X):
        dist = self.build_from_single_tensor(self.params_to_single_tensor(), X[:,:,:-1], X[:,:,-1])
        log_prob = dist.log_prob(y)
        def stable_negative_binomial_log_prob(x, total_count, probs):
            # Clamp to stay in valid ranges
            probs = probs.clamp(1e-6, 1 - 1e-6)
            total_count = total_count.clamp(min=1e-3)
            
            log_factorial = (
                torch.lgamma(x + total_count)
                - torch.lgamma(x + 1)
                - torch.lgamma(total_count)
            )
            log_pmf = (
                log_factorial
                + total_count * torch.log(1 - probs)
                + x * torch.log(probs)
            )
            return log_pmf
        log_prob = stable_negative_binomial_log_prob(y, dist.total_count, dist.probs)
        # Add penalty for random effects
        cov_matrix = self.get_covariance_matrix()
        random_effects = torch.stack([self.b_0, self.b_1], dim=1)
        penalty = -0.5 * torch.mean(
            torch.matmul(random_effects, torch.inverse(cov_matrix)) * random_effects
        )
        
        return torch.mean(log_prob) + penalty


    def ratio_rating(self, X, num_samples=2):
        """
        This is the user-facing call that yields ratio_rating_BD 
        as a normal PyTorch tensor. Internally it calls the 
        custom autograd function.
        """

        # Split into features and time
        time = X[:, :, -1]  # shape: (batch_size, num_locations)
        X_features = X[:, :, :-1]  # shape: (batch_size, num_locations, num_fixed_effects)

        params_single_tensor = self.params_to_single_tensor()
        ratio_rating_BD = RatioRatingFunction.apply(
            params_single_tensor,
            X_features,
            time,
            num_samples,
            self.build_from_single_tensor
        )
        return ratio_rating_BD
    
    def build_from_single_tensor(self, single_tensor, X, time):
        beta_0, beta, b_0, b_1, log_sigma_0, log_sigma_1, rho, siginv_theta = self.single_tensor_to_params(single_tensor)
        fixed_effects = beta_0 + torch.einsum('tli,i->tl', X, beta)
        random_intercepts = b_0.expand(X.shape[0], -1)
        random_slopes = b_1.expand(X.shape[0], -1)
        
        log_mu = fixed_effects + random_intercepts + random_slopes * time

        # Use softplus to ensure mu is positive and grows more slowly
        mu = nn.functional.softplus(log_mu)
        # clamp mu to 50
        mu = torch.clamp(mu, 0, 100)


        # Calculate theta probability
        eps = 1e-6
        theta = torch.clamp(torch.nn.functional.sigmoid(siginv_theta), eps, 1-eps)

        try:
            return NegativeBinomial(total_count=mu, probs=theta)
        except:
            import pdb; pdb.set_trace()
            raise ValueError("Invalid parameters for NegativeBinomial distribution")
        
        
class MixtureOfTruncNormModel(nn.Module):
    def __init__(self, num_components=4, S=12, low=0, high=200):
        super(MixtureOfTruncNormModel, self).__init__()
        self.num_components = num_components
        self.S = S
        self.low=low
        self.high=high
        
        # Initialize the log rates and mixture probabilities as learnable parameters
        self.softplusinv_means = nn.Parameter(torch.rand(num_components)*50, requires_grad=True) 
        self.softplusinv_scales = nn.Parameter(torch.rand(num_components), requires_grad=True) 
        self.mixture_probs = nn.Parameter(torch.rand(S, num_components), requires_grad=True)  

    def params_to_single_tensor(self):
        return torch.cat([param.view(-1) for param in self.parameters()])
    
    def single_tensor_to_params(self, single_tensor):
        softplusinv_means = single_tensor[:self.num_components]
        softplusinv_scales = single_tensor[self.num_components:2*self.num_components]
        mixture_probs = single_tensor[2*self.num_components:].view(self.S, self.num_components)
        return softplusinv_means, softplusinv_scales, mixture_probs
    
    def update_params(self, single_tensor):
        softplusinv_means, softplusinv_scales, mixture_probs = self.single_tensor_to_params(single_tensor)
        self.softplusinv_means.data = softplusinv_means
        self.softplusinv_scales.data = softplusinv_scales
        self.mixture_probs.data = mixture_probs
        return
    
    def build_from_single_tensor(self, single_tensor):
        softplusinv_means, softplusinv_scales, mixture_probs = self.single_tensor_to_params(single_tensor)
        means = torch.nn.functional.softplus(softplusinv_means)
        scales = torch.nn.functional.softplus(softplusinv_scales) + 0.2
        mixture_probs_normalized = torch.nn.functional.softmax(mixture_probs, dim=1)
        categorical_dist = Categorical(mixture_probs_normalized)
        expanded_means = means.expand(self.S, self.num_components)
        expanded_scales = scales.expand(self.S, self.num_components)
        trunc_norm_dist = TruncatedNormal(expanded_means, expanded_scales, self.low, self.high, validate_args=False)
        mixture_dist = MixtureSameFamily(categorical_dist, trunc_norm_dist, validate_args=False)
        return mixture_dist
        
    def forward(self):
        means = torch.nn.functional.softplus(self.softplusinv_means)
        scales = torch.nn.functional.softplus(self.softplusinv_scales) + 0.2
        mixture_probs_normalized = torch.nn.functional.softmax(self.mixture_probs, dim=1)
        categorical_dist = Categorical(mixture_probs_normalized)
        expanded_means = means.expand(self.S, self.num_components)
        expanded_scales = scales.expand(self.S, self.num_components)
        trunc_norm_dist = TruncatedNormal(expanded_means, expanded_scales, self.low, self.high)
        
        # Create the MixtureSameFamily distribution
        mixture_dist = MixtureSameFamily(categorical_dist, trunc_norm_dist,  validate_args=False)
        
        
        
        return mixture_dist
    
    def plot_learned(self, data=None, ax=None):
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
        
        mixture_dist = self.forward()

        samples = mixture_dist.sample((1000,))
        samples = samples.view(-1).detach().numpy()
        # plot the kde only
        sns.kdeplot(samples, bw_adjust=0.5, ax=ax)
        # plot hist of data if present
        if data is not None:
            sns.histplot(data.flatten(), stat='density', bins=50, color='red', alpha=0.5, ax=ax)
        plt.show()
        
        return

    
def torch_bpr_uncurried(y_pred, y_true, K=4, perturbed_top_K_func=None):

    top_K_ids = perturbed_top_K_func(y_pred)
    # Sum over k dim
    top_K_ids = top_K_ids.sum(dim=-2)

    true_top_K_val, _  = torch.topk(y_true, K) 
    denominator = torch.sum(true_top_K_val, dim=-1)
    numerator = torch.sum(top_K_ids * y_true, dim=-1)
    bpr = numerator/denominator

    return bpr

def deterministic_bpr(y_pred, y_true, K=4):

    true_topk = torch.topk(y_true, K)
    pred_topk = torch.topk(y_pred, K)
    # index y_pred with top k indicies from y_true
    true_value_at_pred_topk = torch.gather(y_true, 1, pred_topk.indices)

    numerator = torch.sum(true_value_at_pred_topk, dim=-1)
    denominator = torch.sum(true_topk.values, dim=-1)
    bpr = numerator/denominator
    return bpr
