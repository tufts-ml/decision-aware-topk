import numpy as np
import pandas as pd
import torch
import os

def convert_df_to_3d_array(df):
    # Ensure the DataFrame has a MultiIndex with 'geoid' and 'timestep'
    if not isinstance(df.index, pd.MultiIndex) or set(df.index.names) != {'geoid', 'timestep'}:
        raise ValueError("DataFrame must have a MultiIndex with levels 'geoid' and 'timestep'")

    # Get unique geoids and timesteps, sorted
    geoids = sorted(df.index.get_level_values('geoid').unique())
    timesteps = sorted(df.index.get_level_values('timestep').unique())

    # Create a mapping of geoids to indices
    geoid_to_idx = {geoid: idx for idx, geoid in enumerate(geoids)}

    # Initialize the 3D array
    num_timesteps = len(timesteps)
    num_locations = len(geoids)
    num_features = len(df.columns)
    X = np.zeros((num_timesteps, num_locations, num_features))

    # Fill the 3D array
    for (geoid, timestep), row in df.iterrows():
        t_idx = timesteps.index(timestep)
        g_idx = geoid_to_idx[geoid]
        X[t_idx, g_idx, :] = row.values

    return X, geoids, timesteps

def convert_y_df_to_2d_array(y_df, geoids, timesteps):
    # Ensure the DataFrame has a MultiIndex with 'geoid' and 'timestep'
    if not isinstance(y_df.index, pd.MultiIndex) or set(y_df.index.names) != {'geoid', 'timestep'}:
        raise ValueError("DataFrame must have a MultiIndex with levels 'geoid' and 'timestep'")

    # Initialize the 2D array
    num_timesteps = len(timesteps)
    num_locations = len(geoids)
    y = np.zeros((num_timesteps, num_locations))

    # Create a mapping of geoids to indices
    geoid_to_idx = {geoid: idx for idx, geoid in enumerate(geoids)}

    # Fill the 2D array
    for (geoid, timestep), value in y_df.iloc[:, 0].items():
        t_idx = timesteps.index(timestep)
        g_idx = geoid_to_idx[geoid]
        y[t_idx, g_idx] = value

    return y

def load_data(data_dir, split, device='cpu'):
    X_df = pd.read_csv(os.path.join(data_dir, f'{split}_x.csv'), index_col=[0,1])
    Y_df = pd.read_csv(os.path.join(data_dir, f'{split}_y.csv'), index_col=[0,1])
    X, geoids, timesteps = convert_df_to_3d_array(X_df)
    train_y = convert_y_df_to_2d_array(Y_df, geoids, timesteps)
    time_arr = np.array([timesteps] * len(geoids)).T

    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(train_y, dtype=torch.float32).to(device)
    time = torch.tensor(time_arr, dtype=torch.float32).to(device)

    return X, y, time