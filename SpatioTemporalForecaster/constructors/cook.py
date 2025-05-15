import pandas as pd
import numpy as np
from crane import initialize_from_full_df
import argparse



if __name__ == '__main__':
    """
    Run python crane.py with following:
        temporal_res: year or semiannual

    Returns: dataset with dynamic, static, temporal, and adjacency matrix
    """

    # read data
    parser = argparse.ArgumentParser()
    parser.add_argument('--temporal_res', default='year', choices=['year', 'semiannual'])
    parser.add_argument('--context_size', type=int, default=5)    
    args = vars(parser.parse_args())

    temporal_res = args['temporal_res']
    cook_df = pd.read_csv(f'../../data/cook_county/raw/cook_county_gdf_cleanwithsvi_{temporal_res}.csv')

    # set temporal cols
    if args['temporal_res'] == 'year':
        temporal = ['year']
    elif args['temporal_res'] == 'semiannual':
        temporal = ['year_frac', 'year', 'semiannual', 'season']

    # set lookback window
    context_size = args['context_size']
    dataset_specs = {
        'lookback': context_size,
        'time_length': temporal_res,
        'time_name': 'timestep',
        'space_name': 'geoid',
        'target_name': 'deaths',
        'static': ['INTPTLAT', 'INTPTLON', 'STATEFP','COUNTYFP', 'TRACTCE', 'NAME', 'NAMELSAD', 'MTFCC', 'FUNCSTAT', 'ALAND', 'AWATER'],
        'dynamic': ['svi_theme1_pctile', 'svi_theme2_pctile', 'svi_theme3_pctile', 'svi_theme4_pctile', 'svi_total_pctile', 'pop'],
        'temporal': temporal,
        'lat_name': 'INTPTLAT',
        'long_name': 'INTPTLON',
        'box_length_m': None

    }

    # run analysis
    initialize_from_full_df(cook_df, dataset_specs, type_='cook_county')

