
from functools import partial
import torch

def top_k_onehot_indicator(x, k):

    topk = torch.topk(x, k=k, dim=-1, sorted=False)
    indices = topk.indices
    # convert to k-hot indicator with onehot function
    one_hot = torch.nn.functional.one_hot(indices, num_classes=x.shape[-1]).float()
    #khot = torch.mean(one_hot, dim=-2)
    return one_hot
