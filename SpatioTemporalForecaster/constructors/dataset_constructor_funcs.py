import torch
import pandas as pd
import numpy as np
import argparse
import geopandas as gpd
from pandas.tseries.offsets import DateOffset
from shapely.geometry import Polygon
import os
from datetime import datetime
from collections import namedtuple
import ast
import pickle

"""
Functions that take a T x S x F tensor and output x, y, adjacency matrix
"""


def df_to_tensor(df, temporal_res):
    """
    Converts a dataframe into a torch tensor of shape (T, S, F+1), where:
      - T: number of timesteps (based on the temporal id column)
      - S: number of spatial bins (rows for each timestep)
      - F: number of features (columns ['season_indicator', 'year', 'lat', 'long'])
      - The extra feature is 'last_counts'

    For each spatial tract s at timestep t, the 'last_counts' value is taken from the
    'counts' column at the same spatial tract in timestep t-1. For the first timestep,
    'last_counts' is set to 0.

    Parameters:
      df: the input dataframe (expected to include a 'counts' column)
      temporal_res: a string like 'daily', 'weekly', etc., used to determine the temporal id column name

    Returns:
      A torch tensor of shape (T, S, F+1)
    """
    # Determine the name of the temporal id column using DATE_NAME_TRANSLATOR
    id_col = f"{DATE_NAME_TRANSLATOR[temporal_res]}_id"
    feature_cols = ['season_indicator', 'year', 'lat', 'long']
    
    # Get sorted list of unique timesteps
    timesteps = sorted(df[id_col].unique())
    tensor_list = []
    
    prev_counts_tensor = None  # will hold the sorted 'counts' tensor from the previous timestep
    
    for t in timesteps:
        # Get the sub-dataframe for the current timestep and sort by spatial location
        sub_df = df[df[id_col] == t].copy().sort_values(by=['lat', 'long'])
        
        # Extract the defined features and convert to tensor
        features_tensor = torch.tensor(sub_df[feature_cols].values, dtype=torch.float)
        
        # Determine last_counts: for the first timestep, use zeros; otherwise use the previous timestep's counts
        if prev_counts_tensor is None:
            last_counts = torch.zeros((features_tensor.shape[0], 1), dtype=torch.float)
        else:
            last_counts = prev_counts_tensor
        
        # Concatenate the features with the last_counts column (along the feature dimension)
        combined_tensor = torch.cat([features_tensor, last_counts], dim=1)  # shape: (S, F+1)
        tensor_list.append(combined_tensor)
        
        # Update prev_counts_tensor with the current timestep's counts.
        # Note: We assume that the input dataframe has a 'counts' column.
        current_counts = sub_df['counts'].values
        prev_counts_tensor = torch.tensor(current_counts, dtype=torch.float).unsqueeze(1)
    
    # Stack the list of tensors along the time dimension
    result_tensor = torch.stack(tensor_list, dim=0)  # shape: (T, S, F+1)
    return result_tensor


def df_to_y_tensor(df, temporal_res):
    """
    Converts a dataframe into a torch tensor of shape (T, S), where:
      - T: number of timesteps (determined by the temporal id column)
      - S: number of spatial bins (rows for each timestep)
      
    For each timestep, only the "counts" column is extracted.
    
    Parameters:
      df: the input dataframe.
      temporal_res: a string such as 'daily', 'weekly', etc., used to determine the temporal id column name.
      
    Returns:
      A torch tensor of shape (T, S) representing the counts.
    """
    # Determine the temporal id column using DATE_NAME_TRANSLATOR
    id_col = f"{DATE_NAME_TRANSLATOR[temporal_res]}_id"
    target_col = 'counts'
    
    # Get a sorted list of unique timesteps
    timesteps = sorted(df[id_col].unique())
    
    tensor_list = []
    for t in timesteps:
        # Filter for the current timestep and sort by spatial coordinates for consistency
        sub_df = df[df[id_col] == t].copy().sort_values(by=['lat', 'long'])
        # Convert the counts column to a tensor (shape: (S,))
        sub_tensor = torch.tensor(sub_df[target_col].values, dtype=torch.float)
        tensor_list.append(sub_tensor)
    
    # Stack tensors along the time dimension to get shape (T, S)
    result_tensor = torch.stack(tensor_list, dim=0)
    return result_tensor
    

def compute_adjacency_matrix(df):
    """
    Computes an adjacency matrix for spatial bins using the latitude and longitude columns.
    
    Each entry A[i, j] is defined as:
        A[i, j] = 1 / euclidean_distance(bin_i, bin_j)
    with the convention that for i == j, A[i, j] is set to 1.
    
    Parameters:
      df: DataFrame containing 'lat' and 'long' columns, each row representing a spatial bin.
      
    Returns:
      A torch tensor of shape (S, S) representing the adjacency matrix.
    """
    # Extract coordinates as a tensor of shape (S, 2)
    coords = df[['lat', 'long']].groupby(['lat', 'long']).first().reset_index().values
    coords_tensor = torch.tensor(coords, dtype=torch.float)
    
    # Compute pairwise Euclidean distances
    distances = torch.cdist(coords_tensor, coords_tensor, p=2)
    print(distances.ravel().quantile(0.33))
    print
    
    # Compute reciprocal of distances.
    # Note: division by zero occurs on the diagonal, so we override that next.
    A = 1.0 / (30 * distances + 1)
    A[A < 0.3] = 0
    
    return A


def compute_adjacency_matrix(df, scale_factor=30):
    """
    Computes an adjacency matrix for spatial bins using the latitude and longitude columns.
    
    Each entry A[i, j] is defined as:
        A[i, j] = 1 / euclidean_distance(bin_i, bin_j)
    with the convention that for i == j, A[i, j] is set to 1.
    
    Parameters:
      df: DataFrame containing 'lat' and 'long' columns, each row representing a spatial bin.
      scale_factor: factor by which to normalize distances by. higher means points will count as
        further away from each other. this should change based on the dataset
      
    Returns:
      A torch tensor of shape (S, S) representing the adjacency matrix.
    """
    # Extract coordinates as a tensor of shape (S, 2)
    coords = df[['lat', 'long']].groupby(['lat', 'long']).first().reset_index().values
    coords_tensor = torch.tensor(coords, dtype=torch.float)
    
    # Compute pairwise Euclidean distances
    distances = torch.cdist(coords_tensor, coords_tensor, p=2)
    # Compute reciprocal of distances.
    # Note: division by zero occurs on the diagonal, so we override that next.
    A = 1.0 / (30 * distances + 1)
    A[A < 0.3] = 0
    
    return A