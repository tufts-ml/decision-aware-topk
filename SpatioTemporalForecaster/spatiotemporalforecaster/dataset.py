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


def ensure_dimensions_match(dynamic_feats_TSFd, static_feats_SFs, temp_feats_TFt, adj_SS):
    # Ensure T dimensions match pairwise:
    assert dynamic_feats_TSF.shape[0] == temp_feats_TFt.shape[0], \
        "Mismatch in T dimension: dynamic_feats_TSF.shape[0]={} vs temp_feats_TFt.shape[0]={}".format(
            dynamic_feats_TSF.shape[0], temp_feats_TFt.shape[0])

    # Ensure S dimensions match pairwise:
    assert dynamic_feats_TSF.shape[1] == static_feats_SFs.shape[0], \
        "Mismatch in S dimension: dynamic_feats_TSF.shape[1]={} vs static_feats_SFs.shape[0]={}".format(
            dynamic_feats_TSF.shape[1], static_feats_SFs.shape[0])
    assert dynamic_feats_TSF.shape[1] == adj_SS.shape[0], \
        "Mismatch in S dimension: dynamic_feats_TSF.shape[1]={} vs adj_SS.shape[0]={}".format(
            dynamic_feats_TSF.shape[1], adj_SS.shape[0])
    assert adj_SS.shape[0] == adj_SS.shape[1], \
        "Mismatch in S dimension: adj_SS is not square (shape: {} vs {})".format(
            adj_SS.shape[0], adj_SS.shape[1])

class Dataset:

    def __init__(self, dynamic_feats_TSFd, static_feats_SFs=None, temp_feats_TFt=None, adj_SS=None):

        self.T, self.S, self.Fd = dynamic_feats_TSF.shape

        # check which variables exist
        if not static_feats_SFs:
            static_feats_SFs = torch.Tensor(np.zeros((S, 1)))
        if not temp_feats_TFt:
            temp_feats_TFt = torch.Tensor(np.zeros((T, 1)))
        if not adj_SS:
            adj_SS = torch.Tensor(np.zeros((S, S)))

        self.Fs = static_feats_SFs.shape[1]
        self.Ft = temp_feats_TFt.shape[1]

        # Ensure dimensions match across 
        ensure_dimensions_match(dynamic_feats_TSF, static_feats_SFs, temp_feats_TFt, adj_SS)        

        # Ensure inputs are torch tensors
        dynamic_feats_TSF = set_torch_tensor(dynamic_feats_TSF)
        static_feats_SFs = set_torch_tensor(static_feats_SFs)
        temp_feats_TFt = set_torch_tensor(temp_feats_TFt)
        adj_SS = set_torch_tensor(adj_SS)

        # Initialize features
        self.dynamic_feats_TSF = dynamic_feats_TSF
        self.static_feats_SFs = static_feats_SFs
        self.temp_feats_TFt = temp_feats_TFt
        self.adj_SS = adj_SS
        

    def to_3D(self):
        """
        Returns a TxSxF matrix suitable for modeling
        """

        static_feats_TSFs = self.static_feats_SFs.repeat(T, 1, 1)
        temp_feats_TSFt = self.temp_feats_TFt.repeat(1, S, 1)

        final_3D = torch.cat([self.dynamic_feats_TSF, static_feats_TSFs], dim=2)
        final_3D = torch.cat([final_3D, temp_feats_TSFt], dim=2)
        return final_3D


    def to_2D(self):

        
    


