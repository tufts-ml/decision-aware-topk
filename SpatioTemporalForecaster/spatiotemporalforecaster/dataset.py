"""

"""
import torch
import pandas as pd
import numpy as np


class Dataset:

    def __init__(self, dynamic_feats_TSFd, static_feats_SFs=None, temp_feats_TFt=None, adj_SS=None):
        self.T, self.S, self.Fd = dynamic_feats_TSFd.shape

        # check which variables exist
        if static_feats_SFs is None:
            static_feats_SFs = torch.zeros(self.S, 0)
        if temp_feats_TFt is None:
            temp_feats_TFt = torch.zeros(self.T, 0)
        if adj_SS is None:
            adj_SS = torch.zeros(self.S, self.S)
            # TODO want to set this based on proximity of locations

        self.Fs = static_feats_SFs.shape[1]
        self.Ft = temp_feats_TFt.shape[1]

        # Ensure dimensions match across 
        ensure_dimensions_match(dynamic_feats_TSFd, static_feats_SFs, temp_feats_TFt, adj_SS)        

        # Ensure inputs are torch tensors
        dynamic_feats_TSFd = set_torch_tensor(dynamic_feats_TSFd)
        static_feats_SFs = set_torch_tensor(static_feats_SFs)
        temp_feats_TFt = set_torch_tensor(temp_feats_TFt)
        adj_SS = set_torch_tensor(adj_SS)

        # Initialize features
        self.dynamic_feats_TSFd = dynamic_feats_TSFd
        self.static_feats_SFs = static_feats_SFs
        self.temp_feats_TFt = temp_feats_TFt
        self.adj_SS = adj_SS
        

    @staticmethod
    def ensure_dimensions_match(dynamic_feats_TSFd, static_feats_SFs, temp_feats_TFt, adj_SS):
        # Ensure T dimensions match pairwise:
        assert dynamic_feats_TSFd.shape[0] == temp_feats_TFt.shape[0], \
            "Mismatch in T dimension: dynamic_feats_TSFd.shape[0]={} vs temp_feats_TFt.shape[0]={}".format(
                dynamic_feats_TSFd.shape[0], temp_feats_TFt.shape[0])

        # Ensure S dimensions match pairwise:
        assert dynamic_feats_TSFd.shape[1] == static_feats_SFs.shape[0], \
            "Mismatch in S dimension: dynamic_feats_TSFd.shape[1]={} vs static_feats_SFs.shape[0]={}".format(
                dynamic_feats_TSFd.shape[1], static_feats_SFs.shape[0])
        assert dynamic_feats_TSFd.shape[1] == adj_SS.shape[0], \
            "Mismatch in S dimension: dynamic_feats_TSFd.shape[1]={} vs adj_SS.shape[0]={}".format(
                dynamic_feats_TSFd.shape[1], adj_SS.shape[0])
        assert adj_SS.shape[0] == adj_SS.shape[1], \
            "Mismatch in S dimension: adj_SS is not square (shape: {} vs {})".format(
                adj_SS.shape[0], adj_SS.shape[1])
    
    @staticmethod
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

            

    def to_3D(self):
        """
        Returns a TxSxF matrix suitable for modeling
        """
        # print('final 3D 0', self.dynamic_feats_TSFd.shape)
        # print('static feats 0', self.static_feats_SFs.shape)

        # check static feats
        if not (self.static_feats_SFs.shape == (self.S, 0) and self.static_feats_SFs == torch.zeros(self.S, 0)):
            static_feats_TSFs = self.static_feats_SFs.unsqueeze(0)
            static_feats_TSFs = static_feats_TSFs.repeat(self.T, 1, 1)
            print('static feats 1', static_feats_TSFs.shape)
            final_3D = torch.cat([self.dynamic_feats_TSFd, static_feats_TSFs], dim=2)
        else:
            final_3D = self.dynamic_feats_TSFd.clone()

        # print('final 3D 1', final_3D.shape)
        # print('temp feats 0', self.temp_feats_TFt.shape)

        # check temp feats
        if not (self.temp_feats_TFt.shape == (self.T, 0) and self.temp_feats_TFt == torch.zeros(self.T, 0)):
            temp_feats_TSFt = self.temp_feats_TFt.unsqueeze(1)
            temp_feats_TSFt = temp_feats_TSFt.repeat(1, self.S, 1)
            # print('temp feats 1', temp_feats_TSFt.shape)
            final_3D = torch.cat([final_3D, temp_feats_TSFt], dim=2)

        # print('final 3D 2', final_3D.shape)

        return final_3D


    def to_2D(self):
        """
        Returns (TxS)xF matrix suitable for modeling
        """

        return self.to_3D().flatten(start_dim=0, end_dim=1)

        
    


