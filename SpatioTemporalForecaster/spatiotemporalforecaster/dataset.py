"""

"""
import torch
import pandas as pd
import numpy as np

def set_torch_tensor(df):
    """
    Converts a dataframe to a torch tensor.
    Parameters:
        df (torch.Tensor or pd.DataFrame or np.array): The input dataframe.
    Returns:
        torch.Tensor: The converted torch tensor.
    Raises:
        ValueError: If the input is not a recognized type (torch Tensor, Pandas DataFrame, or Numpy array).
    """
    if isinstance(df, torch.Tensor):
        return df
    elif isinstance(df, pd.DataFrame):
        return torch.from_numpy(df.values)
    elif isinstance(df, type(np.array([]))):
        return torch.from_numpy(df)
    else:
        raise ValueError('Unrecognized input. Please only use torch Tensors, Pandas DataFrames, or Numpy arrays')


def ensure_dimensions_match(dynamic_feats_TSF, static_feats_SF, temp_feats_TF, adj_SS):
    # Ensure T dimensions match pairwise:
    assert dynamic_feats_TSF.shape[0] == temp_feats_TF.shape[0], \
        "Mismatch in T dimension: dynamic_feats_TSF.shape[0]={} vs temp_feats_TF.shape[0]={}".format(
            dynamic_feats_TSF.shape[0], temp_feats_TF.shape[0])

    # Ensure S dimensions match pairwise:
    assert dynamic_feats_TSF.shape[1] == static_feats_SF.shape[0], \
        "Mismatch in S dimension: dynamic_feats_TSF.shape[1]={} vs static_feats_SF.shape[0]={}".format(
            dynamic_feats_TSF.shape[1], static_feats_SF.shape[0])
    assert dynamic_feats_TSF.shape[1] == adj_SS.shape[0], \
        "Mismatch in S dimension: dynamic_feats_TSF.shape[1]={} vs adj_SS.shape[0]={}".format(
            dynamic_feats_TSF.shape[1], adj_SS.shape[0])
    assert adj_SS.shape[0] == adj_SS.shape[1], \
        "Mismatch in S dimension: adj_SS is not square (shape: {} vs {})".format(
            adj_SS.shape[0], adj_SS.shape[1])

    # Ensure F dimensions match pairwise:
    assert dynamic_feats_TSF.shape[2] == static_feats_SF.shape[1], \
        "Mismatch in F dimension: dynamic_feats_TSF.shape[2]={} vs static_feats_SF.shape[1]={}".format(
            dynamic_feats_TSF.shape[2], static_feats_SF.shape[1])
    assert dynamic_feats_TSF.shape[2] == temp_feats_TF.shape[1], \
        "Mismatch in F dimension: dynamic_feats_TSF.shape[2]={} vs temp_feats_TF.shape[1]={}".format(
            dynamic_feats_TSF.shape[2], temp_feats_TF.shape[1])


class Dataset:

    def __init__(self, dynamic_feats_TSF, static_feats_SF, temp_feats_TF, adj_SS):

        # Ensure dimensions match across 
        ensure_dimensions_match(dynamic_feats_TSF, static_feats_SF, temp_feats_TF, adj_SS)        

        # Ensure inputs are torch tensors
        dynamic_feats_TSF = set_torch_tensor(dynamic_feats_TSF)
        static_feats_SF = set_torch_tensor(static_feats_SF)
        temp_feats_TF = set_torch_tensor(temp_feats_TF)
        adj_SS = set_torch_tensor(adj_SS)

        # Initialize features
        self.T, self.S, self.F = dynamic_feats_TSF.shape
        self.dynamic_feats_TSF = dynamic_feats_TSF
        self.static_feats_SF = static_feats_SF
        self.temp_feats_TF = temp_feats_TF
        self.adj_SS = adj_SS

    def to_3D(self):
        pass

    def to_2D(self):
        
        
    


