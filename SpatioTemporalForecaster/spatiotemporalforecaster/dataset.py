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
import json


class Dataset:
    """
    Class to store data for spatiotemporal modeling
    """

    def df_to_tensor_dynamic(self, df, feature_cols, lookback=5, time_name='bimonth_id', space_name='geoid', target_name='counts'):

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

            # Create current counts tensor
            current_counts = sub_df[target_name].values
            current_counts_tensor = torch.tensor(current_counts, dtype=torch.float).unsqueeze(1)

            # Concatenate features with lag tensors
            combined_tensor = torch.cat([current_counts_tensor, features_tensor, lag_tensor], dim=1)  # shape: (S, F + lookback)
            tensor_list.append(combined_tensor)

            # Update prev_counts_list with current counts tensor
            prev_counts_list.insert(0, current_counts_tensor)
            if len(prev_counts_list) > lookback:
                prev_counts_list.pop()

        # Stack along time dimension
        result_tensor = torch.stack(tensor_list, dim=0)  # shape: (T, S, F + lookback)
        return result_tensor

    def df_to_tensor_static(self, df, feature_cols, space_name='geoid'):
        return df.groupby(space_name).first().reset_index()

    def df_to_tensor_temporal(self, df, feature_cols, time_name):
        return df.groupby(time_name).first().reset_index()

    """
    Functions that take a T x S x F tensor and output x, y, adjacency matrix
    """
    def df_to_tensor(self, df, type_='dynamic', lookback=5, time_name='bimonth', space_name='geoid', target_name='counts', static=None, dynamic=None, temporal=None, latlong=True, box_length_m=500, **kwargs):
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
            func = self.df_to_tensor_dynamic

        elif type_ == 'static':

            df = df[[space_name] + static]
            feature_cols = static
            args = {'space_name': space_name} 
            func = self.df_to_tensor_static
        
        elif type_ == 'temporal':
        
            df = df[[time_name] + temporal]
            feature_cols = temporal
            args = {'time_name': time_name} 
            func = self.df_to_tensor_temporal

        else: 
            raise ValueError('dataset type must be dynamic, static, or temporal')
    
        print('features', df.columns)
        return func(df, feature_cols, **args)


    def compute_adjacency_matrix(self, df, dist_sensitivity=30, **dataset_specs):
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
        
        lat = dataset_specs['lat_name']  
        long = dataset_specs['long_name']  

        if lat and long:

            # Extract coordinates as a tensor of shape (S, 2)
            coords = df[[lat, long]].groupby([lat, long]).first().reset_index().values
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


    def to_disk(self, filename='aerial_surv'):
        """
        Loads datasets to disk. If data was originally initialized with column names, path to data is based on those
        """
        if self.dataset_specs:
            time_name = self.dataset_specs['time_name']
            time_length = self.dataset_specs['time_length']
            box_length_m = self.dataset_specs['box_length_m']
            if box_length_m: 
                path_to_final_data = f'../../data/{filename}/model-ready/{time_length}_{box_length_m}M'
            else:
                path_to_final_data = f'../../data/{filename}/model-ready/{time_length}'
        else:
            path_to_final_data = f'../../data/{filename}/model-ready/{time_length}'

        if not os.path.exists(path_to_final_data):
            os.makedirs(path_to_final_data)
        if not os.path.exists(f"{path_to_final_data}/dynamic"):
            os.makedirs(f"{path_to_final_data}/dynamic")

        # save data
        for arr_slice in range(self.dynamic_feats_TSFd.shape[0]):
            np.savetxt(f'{path_to_final_data}/dynamic/tstep_{arr_slice}.csv', self.dynamic_feats_TSFd[arr_slice, :, :].numpy(), delimiter=',')

        np.savetxt(f'{path_to_final_data}/static.csv', self.static_feats_SFs.numpy(), delimiter=',')
        np.savetxt(f'{path_to_final_data}/temporal.csv', self.temp_feats_TFt.numpy(), delimiter=',')
        np.savetxt(f'{path_to_final_data}/adjacency.csv', self.adj_SS.numpy(), delimiter=',')
        with open(f"{path_to_final_data}/dataset_specs.json", "w") as f:
            json.dump(self.dataset_specs, f, indent=4)

        print(f'Data loaded to {path_to_final_data}')


    def initialize_from_full_df(self, full_df, dataset_specs, type_='aerial_surv'):
        """
                dataset_specs = {
                        'lookback':
                        'time_name': 
                        'space_name': 
                        'target_name':
                        'static': 
                        'dynamic': 
                        'temporal':
                        'lat_name': 
                        'long_name':
                        'box_length_m'
                    }
        """

        print(full_df)
        print(full_df.columns)

        # load tensors to class
        self.dynamic_feats_TSFd = self.df_to_tensor(full_df, type_='dynamic', **dataset_specs)
        self.static_feats_SFs = torch.tensor(self.df_to_tensor(full_df, type_='static', **dataset_specs).values)
        self.temp_feats_TFt = torch.tensor(self.df_to_tensor(full_df, type_='temporal', **dataset_specs).values)

        print

        # TODO add dist_sensitivity as user argument
        self.adj_SS = self.compute_adjacency_matrix(full_df, dist_sensitivity=30, **dataset_specs)
        self.dataset_specs = dataset_specs

        self.T, self.S, self.Fd = self.dynamic_feats_TSFd.shape
        self.Fs = self.static_feats_SFs.shape[1]
        self.Ft = self.temp_feats_TFt.shape[1]

    @staticmethod
    def validate_inputs(path_to_folder=None,
                    full_df=None, dataset_specs=None,
                    dynamic_feats_TSFd=None, static_feats_SFs=None,
                    temp_feats_TFt=None, adj_SS=None):
        groups = [
            {'path_to_folder': path_to_folder},
            {'full_df': full_df, 'dataset_specs': dataset_specs},
            {'dynamic_feats_TSFd': dynamic_feats_TSFd,
            'static_feats_SFs': static_feats_SFs,
            'temp_feats_TFt': temp_feats_TFt,
            'adj_SS': adj_SS}
        ]

        # Count how many groups are fully non-None
        valid_group_count = sum(
            all(val is not None for val in group.values())
            for group in groups
        )

        if valid_group_count != 1:
            raise ValueError(
                "Exactly one of the following sets must be fully defined (i.e., none of the values in the set can be None):\n"
                "1. {path_to_folder}\n"
                "2. {full_df, dataset_specs}\n"
                "3. {dynamic_feats_TSFd, static_feats_SFs, temp_feats_TFt, adj_SS}"
            )

    import os, json, numpy as np, torch


    @staticmethod
    def from_disk(path_to_final_data):
        """
        Reads the files under `path_to_final_data` (the folder containing
        model-ready/static.csv, temporal.csv, adjacency.csv, dynamic/tstep_*.csv
        and dataset_specs.json) and returns a new Dataset instance.
        """
        # --- load specs ---
        specs_file = os.path.join(path_to_final_data, 'dataset_specs.json')
        if not os.path.isfile(specs_file):
            dataset_specs = None
        else:
            with open(specs_file, 'r') as f:
                dataset_specs = json.load(f)

        # --- load dynamic features ---
        dyn_dir = os.path.join(path_to_final_data, 'dynamic')
        if not os.path.isdir(dyn_dir):
            raise FileNotFoundError(f"Couldn’t find dynamic array folder at {dyn_dir}")

        # read in dynamic file
        # pick up all tstep_N.csv in order
        tstep_files = [
            fname for fname in os.listdir(dyn_dir)
            if fname.startswith('tstep_') and fname.endswith('.csv')
        ]
        tstep_files.sort(key=lambda fn: int(fn.split('_')[1].split('.')[0]))
        dyn_slices = []
        for fname in tstep_files:
            arr = np.loadtxt(os.path.join(dyn_dir, fname), delimiter=',')
            dyn_slices.append(torch.from_numpy(arr))
        dynamic_feats_TSFd = torch.stack(dyn_slices, dim=0)

        # --- load the remaining arrays ---
        static_feats_SFs = torch.from_numpy(
            np.loadtxt(os.path.join(path_to_final_data, 'static.csv'), delimiter=',')
        )
        temp_feats_TFt = torch.from_numpy(
            np.loadtxt(os.path.join(path_to_final_data, 'temporal.csv'), delimiter=',')
        )
        adj_SS = torch.from_numpy(
            np.loadtxt(os.path.join(path_to_final_data, 'adjacency.csv'), delimiter=',')
        )
        
        return dynamic_feats_TSFd, static_feats_SFs, temp_feats_TFt, adj_SS, dataset_specs


    def __init__(self, path=None, full_df=None, dataset_specs=None, dynamic_feats_TSFd=None, static_feats_SFs=None, temp_feats_TFt=None, adj_SS=None, **kwargs):

        """
        User can initialize with:
            - pre-loaded dataframes (dynamic_feats_TSFd, static_feats_SFs, temp_feats_TFt, adj_SS)
        OR:
            - user can upload a dataframe with the following form
                (T x S) x F
                and a dict with
                    TODO fill
                    dataset_specs = {
                        'lookback':
                        'time_name': 
                        'space_name': 
                        'target_name':
                        'static': 
                        'dynamic': 
                        'temporal': 
                        'latlong': 
                        'box_length_m': 
                    }
        OR:
             user can add a path to a folder they previously loaded the data onyo
        """

        self.validate_inputs(path,
                    full_df, dataset_specs,
                    dynamic_feats_TSFd, static_feats_SFs,
                    temp_feats_TFt, adj_SS)

        if dataset_specs:
            self.initialize_from_full_df(full_df, dataset_specs)
        
        elif path:

            self.dynamic_feats_TSFd, self.static_feats_SFs, self.temp_feats_TFt, self.adj_SS, self.dataset_specs = Dataset.from_disk(path)
            self.T, self.S, self.Fd = self.dynamic_feats_TSFd.shape
            self.Fs = self.static_feats_SFs.shape[1]
            self.Ft = self.temp_feats_TFt.shape[1]

        else:

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
            self.dataset_specs = None
        

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

        
    


