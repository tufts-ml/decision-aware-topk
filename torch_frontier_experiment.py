import os
import torch
import numpy as np

from distributions import QuantizedNormal
from torch_models import  MixtureOfTruncNormModel
#from torch_perturb.torch_pert_topk import PerturbedTopK
from torch_perturb.perturbations import perturbed
from functools import partial
from metrics import top_k_onehot_indicator
from torch_training import train_epoch

def main(step_size=None, epochs=None, bpr_weight=None,
         nll_weight=None, seed=None, init_idx=None, outdir=None, epsilon=None,
           num_components=None, perturbed_noise=None, initialization=None):


    # total timepoints
    T= 500
    S=7
    K=5
    device = 'cpu'

    dist_S = [QuantizedNormal(10, 0.3),
            QuantizedNormal(20, 0.3),
            QuantizedNormal(30, 0.3),
            QuantizedNormal(40, 0.3),
            QuantizedNormal(50, 0.3),
            QuantizedNormal(60, 0.3),
            QuantizedNormal(100, 0.3)]


    train_y_TS = np.zeros((T, S))
    for s, dist in enumerate(dist_S):
        random_state = np.random.RandomState(10000 * seed + s*123456)
        train_y_TS[:, s] = dist.rvs(size=T, random_state=random_state)

    model = MixtureOfTruncNormModel(num_components=num_components, S=S, low=0, high=150)

    if init_idx is not None:
        # Reproducibly, randomly generate some numbers using a numpy rng
        init_rng = np.random.RandomState(1989)
        # generate 20 sets of 2 floats between 0.5 and 60
        all_means = init_rng.uniform(8, 102, (20, num_components))
        # generate 20 sets of 2 floats between 0.25 and 6
        all_scales = init_rng.uniform(0.25, 8, (20, num_components))
        # generate 20 lists length S containing lists length 2 which sum to 1
        all_mix_weights = init_rng.dirichlet([0.5]*num_components, (20,S))

        softinv_means = torch.tensor(all_means[init_idx]) + torch.log(-torch.expm1(torch.tensor(-all_means[init_idx])))
        softinv_scales = torch.tensor(all_scales[init_idx]) - 0.2 + torch.log(-torch.expm1(torch.tensor(-all_scales[init_idx] + 0.2)))
        mix_weights = torch.log(1e-13 + torch.tensor(all_mix_weights[init_idx]))
        model.update_params(torch.cat([softinv_means, softinv_scales, mix_weights.view(-1)]))

    if initialization == 'bpr':
        means = torch.tensor([1, 10])
        softinv_means = means + torch.log(-torch.expm1(-means))
        scales = torch.tensor([0.25, 0.25])
        softinv_scales = scales - 0.2 + torch.log(-torch.expm1(-scales + 0.2)) 
        mix_weights = torch.log(1e-13 + torch.tensor(
                                        [[1,0]*20+[0,1]*3]))
        model.update_params(torch.cat([softinv_means, softinv_scales, mix_weights.view(-1)]))
    elif initialization == 'nll':
        means = torch.tensor([10, 37])
        softinv_means = means + torch.log(-torch.expm1(-means))
        scales = torch.tensor([0.3, 5])
        softinv_scales = scales - 0.2 + torch.log(-torch.expm1(-scales + 0.2)) 
        mix_weights = torch.log(1e-13 + torch.tensor(
                                        [[1,0]*10+[0,1]*13]))
        model.update_params(torch.cat([softinv_means, softinv_scales, mix_weights.view(-1)]))


    model.to(device)
    model.float()
    


    optimizer = torch.optim.Adam(model.parameters(), lr=step_size)

    M_score_func =  100
    M_action = 100
    train_T = train_y_TS.shape[0]
    #perturbed_top_K_func = PerturbedTopK(k=K, sigma=perturbed_noise)
     # Setup top-k function
    top_k_func = partial(top_k_onehot_indicator, k=K)
    perturbed_top_K_func = perturbed(top_k_func, sigma=perturbed_noise, num_samples=M_action, device=device)
    losses, bprs, nlls = [], [], []
    for epoch in range(epochs):
        print(f'EPOCH: {epoch}')
        loss, bpr, nll, model = train_epoch(model, optimizer, K, epsilon, train_T, M_score_func, M_action, train_y_TS, perturbed_top_K_func, bpr_weight, nll_weight)
        losses.append(loss)
        bprs.append(bpr)
        nlls.append(nll)
    
        # save everything to outdir every 100 epochs
        if epoch % 100 == 0:
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            torch.save(model.state_dict(), f'{outdir}/model.pth')
            torch.save(optimizer.state_dict(), f'{outdir}/optimizer.pth')
            torch.save(losses, f'{outdir}/losses.pth')
            torch.save(bprs, f'{outdir}/bprs.pth')
            torch.save(nlls, f'{outdir}/nlls.pth')


if __name__ ==  "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--step_size", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--bpr_weight", type=float, default=1.0)
    parser.add_argument("--nll_weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=360)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--epsilon", type=float, default=0.55)
    parser.add_argument("--num_components", type=int, default=4)
    parser.add_argument("--perturbed_noise", type=float, default=0.05)
    parser.add_argument("--initialization", type=str, required=False)
    parser.add_argument("--init_idx", type=int, required=False)

    args = parser.parse_args()
    # call main with args as keywrods
    main(**vars(args))