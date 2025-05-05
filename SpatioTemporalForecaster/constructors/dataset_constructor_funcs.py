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


# defining date range
DATE_RANGE_TRANSLATOR = {  
    'daily': 'D',
    'weekly': 'W',
    'biweekly': '2W',
    'monthly': 'ME',
    '2monthly': '2ME',
    '3monthly': '3ME'
}
# how much temporal buffer to give based on resolution
DATE_OFFSET_TRANSLATOR = {  
    'daily': 1,
    'weekly': 7,
    'biweekly': 14,
    'monthly': 30,
    '2monthly': 60,
    '3monthly': 90
}
# naming the temporal column
DATE_NAME_TRANSLATOR = {  
    'daily': 'day',
    'weekly': 'week',
    'biweekly': 'biweek',
    'monthly': 'month',
    '2monthly': 'bimonth',
    '3monthly': 'trimonth',
    'seasonal': 'season'
}

MONTH_PAIRS_TRANSLATOR = {
    '2monthly': ["02-28", "04-30", "10-20", "12-25"],
    '3monthly': ["01-31", "04-30", "10-20"]
}



MAP_SIZE_TRANSLATOR = {
    'medium': {
        'y_left_lower_line': 0,
        'y_right_lower_line': 0.45,
        'y_left_upper_line': 0.35,
        'y_right_upper_line': 0.95
    },
    'small': {
        'y_left_lower_line': 0.06,
        'y_right_lower_line': 0.72,
        'y_left_upper_line': 0.3,
        'y_right_upper_line': 0.5
    }
}

SEASONAL_TRANSLATOR = {
    9: 0,
    10: 1,
    11: 2,
    12: 3,
    1: 4,
    2: 5,
    3: 6, 
    4: 7
}

# meters per degree lat or long
METERS_PER_DEGREE = 111111


def df_to_tensor_dynamic(df, feature_cols, lookback=5, time_name='bimonth_id', space_name='geoid', target_name='counts'):

    # Determine the name of the temporal id column
    id_col = time_name

    # Get sorted list of unique timesteps
    timesteps = sorted(df[id_col].unique())

    tensor_list = []
    prev_counts_list = []  # list to hold counts tensors from previous timesteps

    for t in timesteps:
        sub_df = df[df[id_col] == t].copy().sort_values(by=space_name)
        features_tensor = torch.tensor(sub_df[feature_cols].values, dtype=torch.float)

        # Build list of lagged count tensors
        lag_tensors = []
        for k in range(1, lookback + 1):
            if len(prev_counts_list) >= k:
                lag_tensors.append(prev_counts_list[k - 1])
            else:
                # Backfill missing lags with NaN for the first lookback timesteps
                lag_tensors.append(torch.full((features_tensor.shape[0], 1), float('nan'), dtype=torch.float))
        lag_tensor = torch.cat(lag_tensors, dim=1)  # shape: (S, lookback)

        # Concatenate features with lag tensors
        combined_tensor = torch.cat([features_tensor, lag_tensor], dim=1)  # shape: (S, F + lookback)
        tensor_list.append(combined_tensor)

        # Update prev_counts_list with current counts tensor
        current_counts = sub_df[target_name].values
        current_counts_tensor = torch.tensor(current_counts, dtype=torch.float).unsqueeze(1)
        prev_counts_list.insert(0, current_counts_tensor)
        if len(prev_counts_list) > lookback:
            prev_counts_list.pop()

    # Stack along time dimension
    result_tensor = torch.stack(tensor_list, dim=0)  # shape: (T, S, F + lookback)
    return result_tensor


def df_to_tensor_static(df, feature_cols, space_name='geoid'):

  return df.groupby(space_name).first()

def df_to_tensor_temporal(df, feature_cols, time_name):

  return df.groupby(time_name).first()


"""
Functions that take a T x S x F tensor and output x, y, adjacency matrix
"""
def df_to_tensor(df, type_='dynamic', lookback=5, time_name='bimonth_id', space_name='geoid', target_name='counts', static=None, dynamic=None, temporal=None, latlong=True, box_length_m=500):
    """
    Converts a dataframe into a torch tensor of shape (T, S, F + lookback), where:
      - T: number of timesteps (based on the temporal id column)
      - S: number of spatial bins (rows for each timestep)
      - F: number of features (columns defined by 'features' param, default ['season_indicator', 'year', 'lat', 'long'])
      - lookback: number of lagged counts to include, from t-1 to t-lookback

    For each spatial tract s at timestep t, the lagged count columns are taken from the
    'target_name' column at the same spatial tract in previous timesteps. Missing lags are filled with NaN.

    Parameters:
      df: the input dataframe (expected to include the target_name column)
      lookback: number of past timesteps to include as features
      time_name: column name for temporal id
      space_name: column name for spatial id
      target_name: column name for counts
      features: list of feature column names; if None, defaults to ['season_indicator', 'year', 'lat', 'long'] depending on latlong flag
      latlong: whether to include 'lat' and 'long' in default features

    Returns:
      A torch tensor of shape (T, S, F + lookback)
    """    

    if type_ == 'dynamic':

      df = df[[space_name, time_name, target_name] + dynamic]
      feature_cols = dynamic
      args = {'lookback': lookback, 'time_name': time_name, 'space_name': space_name, 'target_name': target_name} # TODO add
      func = df_to_tensor_dynamic

    elif type_ == 'static':

      df = df[[space_name] + static]
      feature_cols = static
      args = {'space_name': space_name} 
      func = df_to_tensor_static
    
    elif type_ == 'temporal':
      
      df = df[[time_name] + temporal]
      feature_cols = temporal
      args = {'time_name': time_name} 
      func = df_to_tensor_temporal

    else: 
      raise ValueError('dataset type must be dynamic, static, or temporal')
  
    print('features', df.columns)
    return func(df, feature_cols, **args)



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
    

def compute_adjacency_matrix(df, dist_sensitivity=30, **dataset_specs):
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
  
  if 'latlong' in dataset_specs:
    latlong = dataset_specs['latlong']
  else:
    latlong = None    

  if latlong:

    # Extract coordinates as a tensor of shape (S, 2)
    coords = df[['lat', 'long']].groupby(['lat', 'long']).first().reset_index().values
    coords_tensor = torch.tensor(coords, dtype=torch.float)

    # Compute pairwise Euclidean distances
    distances = torch.cdist(coords_tensor, coords_tensor, p=2)
    print(distances.ravel().quantile(0.33))

    # Compute reciprocal of distances.
    # Note: division by zero occurs on the diagonal, so we override that next.
    A = 1.0 / (dist_sensitivity * distances + 1)

    return A
  
  else:

    raise ValueError('latitude and longitude columns must exist')

