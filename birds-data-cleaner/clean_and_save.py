import pandas as pd
from pandas.tseries.offsets import DateOffset
from pandas import IndexSlice as idx
from shapely.geometry import Polygon
import numpy as np
import geopandas as gpd
import os
import ast
from datetime import datetime
from collections import namedtuple
import sys

"""
OPTIONS FOR PARAMETERS:
years_through_2011: numeric, integer
temporal_res = ['daily', 'weekly', 'biweekly', 'monthly']
box_length_m: numeric, meters
complete_idx_square: bool. --> this dataset does not come with the square completed, 
    so can switch this to true to complete square. Default is false.
keep_geometry_col: bool. --> saves a lot of space if this is set to false. default is true. 
"""

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
    10: 1,
    12: 2,
    2: 3
}

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

# meters per degree lat or long
METERS_PER_DEGREE = 111111

def cut_gpd_water(gpdf, x_left=0, y1_up=(1/3), x_right=0, y2_up=0.5, less=True):

    gpdf['centroids'] = gpdf['geometry_col'].centroid

    bound = gpdf.total_bounds
    x_lower = bound[0]
    y_lower = bound[1]
    x_upper = bound[2]
    y_upper = bound[3]

    x1 = x_lower
    y1 = y_lower + (y_upper - y_lower) * y1_up
    x2 = x_upper
    y2 = y_lower + (y_upper - y_lower) * y2_up

    slope = (y2 - y1) / (x2 - x1)
    
    if less:
        return gpdf[slope * (gpdf['centroids'].x - x1) + y1 - gpdf['centroids'].y <= 0].drop(['centroids'], axis=1)
    
    return gpdf[slope * (gpdf['centroids'].x - x1) + y1 - gpdf['centroids'].y >= 0].drop(['centroids'], axis=1)



def get_aransas_box_bounds():
    """
    If we don't have access to aransas box bounds recreate them
    """

    # TODO untested
    # Set spacing and generate the grid
    aransas_df = pd.read_csv('raw-data/asurv_1950_to_2011/WHCR_Aerial_Observations_1950_2011.txt', encoding='latin1', sep='\t')
    aransas_gdf = gpd.GeoDataFrame(aransas_df, geometry=gpd.points_from_xy(aransas_df.X, aransas_df.Y), crs='EPSG:26914')
    bbox = aransas_gdf.total_bounds
    box_length_degrees = box_length_m / METERS_PER_DEGREE
    grid_gdf = generate_grid(bbox, box_length_degrees, crs=aransas_gdf.crs)
    return grid_gdf.total_bounds

# Function to generate a grid of boxes
def generate_grid(bbox, spacing, crs):
    """
    Generate box grid based on min x, min y, max x, and max y (LONG/LAT)
    Spacing: Space between each box in degrees
    Crs: Coordinate reference system
    """
    METERS_PER_DEGREE = 111111

    if crs.to_string() == 'EPSG:26914':
        spacing = spacing * METERS_PER_DEGREE

    minx, miny, maxx, maxy = bbox
    x_coords = np.arange(minx, maxx, spacing)
    y_coords = np.arange(miny, maxy, spacing)

    grid = []
    for x in x_coords:
        for y in y_coords:
            grid.append(Polygon([(x, y), (x + spacing, y), (x + spacing, y + spacing), (x, y + spacing), (x, y)]))
    return gpd.GeoDataFrame({'geometry': grid}, crs=crs)


def points_to_boxes(gdf, study, temporal_res, box_length_m, keep_geometry_col, complete_idx_square, map_size='small', context_size=5, **kwargs):
    
    """
    Given gdf with x;y;count;tstep, assign counts to boxes 
    """

    if 'years_through_2011' in kwargs.keys():
        years_through_2011 = kwargs['years_through_2011']
    if 'years_cut_from_back' in kwargs.keys():
        years_cut_from_back = kwargs['years_cut_from_back']
    
    all_gdfs = []
    for tstep in np.sort(gdf[f'{DATE_NAME_TRANSLATOR[temporal_res]}_name'].unique()):
        
        filtered_gdf = gdf[gdf[f'{DATE_NAME_TRANSLATOR[temporal_res]}_name'] == tstep]
        # Read in bounding box from data folder
        with open("boxes_total_bounds.txt", "r") as file:
            bounds = file.read()
        
        min_x, min_y, max_x, max_y = ast.literal_eval(bounds)

        # MAKE SURE we are in the CRS that measures by meters, not lat/long
        assert (filtered_gdf.crs.to_string() == 'EPSG:26914')
        grid_cells = []
        for x in np.arange(min_x, max_x, box_length_m):
            for y in np.arange(min_y, max_y, box_length_m):
                grid_cells.append(Polygon([
                    (x, y),
                    (x + box_length_m, y),
                    (x + box_length_m, y + box_length_m),
                    (x, y + box_length_m)
                ]))

        # Create a GeoDataFrame for the grid
        full_grid = gpd.GeoDataFrame({'geometry': grid_cells}, crs=filtered_gdf.crs)

        # Perform a spatial join to count the number of points in each grid cell
        joined = gpd.sjoin(filtered_gdf, full_grid, how='left', predicate='within')
        counts = joined.groupby('index_right').agg({'count': 'sum'})

        # Add the counts to the grid GeoDataFrame
        full_grid['counts'] = counts
        full_grid[f'{DATE_NAME_TRANSLATOR[temporal_res]}_id'] = filtered_gdf[f'{DATE_NAME_TRANSLATOR[temporal_res]}'].unique()[0]
        full_grid[f'{DATE_NAME_TRANSLATOR[temporal_res]}_name'] = tstep

        # add unique date
        gdf[gdf[f'{DATE_NAME_TRANSLATOR[temporal_res]}_name'] == tstep]['date'].unique()[0]

        full_grid['counts'] = full_grid['counts'].fillna(0)

        # add indicator
        full_grid['month_indicator'] = SEASONAL_TRANSLATOR[datetime.strptime(tstep.split('_')[0], '%Y-%m-%d').month]
        full_grid['year'] = datetime.strptime(tstep.split('_')[0], '%Y-%m-%d').year
        
        print(f"unique counts for {tstep}: {full_grid['counts'].unique()}")
        all_gdfs.append(full_grid)
    
    combined_gdf = pd.concat(all_gdfs)
    print('shape of combined GDF', combined_gdf.shape[0])
    combined_gdf = combined_gdf[combined_gdf[f"{DATE_NAME_TRANSLATOR[temporal_res]}_id"] >= 148]
    
    # create lat long columns and save to CSV
    combined_gdf.set_index([f'{DATE_NAME_TRANSLATOR[temporal_res]}_id', 'geometry'], drop=True, inplace=True)
    
    combined_gdf['geometry_col'] = combined_gdf.index.get_level_values(1)
    combined_gdf = combined_gdf.set_geometry('geometry_col')
    
    centers = combined_gdf.geometry.centroid
    centers_latlong = centers.to_crs('EPSG:4326')
    combined_gdf['lat'] = centers_latlong.y
    combined_gdf['long'] = centers_latlong.x

    if map_size != 'full':

        print(combined_gdf)
        y_left_lower_line = MAP_SIZE_TRANSLATOR[map_size]['y_left_lower_line']
        y_right_lower_line = MAP_SIZE_TRANSLATOR[map_size]['y_right_lower_line']
        y_left_upper_line = MAP_SIZE_TRANSLATOR[map_size]['y_left_upper_line']
        y_right_upper_line = MAP_SIZE_TRANSLATOR[map_size]['y_right_upper_line']

        combined_gdf = cut_gpd_water(cut_gpd_water(combined_gdf, y1_up=y_left_lower_line, y2_up=y_right_lower_line, less=True), y1_up=y_left_upper_line, y2_up=y_right_upper_line, less=False)


    if not os.path.exists(f'data_dir/{temporal_res}_ctxtSize{context_size}_{map_size}Map'):
        os.makedirs(f'data_dir/{temporal_res}_ctxtSize{context_size}_{map_size}Map') 

    # Make "README" of sorts for each dataset
    with open(f'data_dir/{temporal_res}_ctxtSize{context_size}_{map_size}Map/EDA.txt', 'w') as file:
        
        file.write(f'This file contains basic information about the dataset.\n\n')
        
        # basic info
        # file.write(f"Is the square completed for time-space indices? {complete_idx_square}\n")
        # file.write(f"Is the geometry column present? {keep_geometry_col}\n")

        # total number of boxes
        file.write(f"TOTAL # OF BOXES: {len(combined_gdf['geometry_col'].unique())}\n")

        print(combined_gdf.columns)
        # total number of timesteps + how many years
        file.write(f"TOTAL # OF TIMESTEPS: {len(combined_gdf[f'{DATE_NAME_TRANSLATOR[temporal_res]}_name'])}\n")

        # min date, max date
        file.write(f"MIN DATE: {np.sort(combined_gdf[f'{DATE_NAME_TRANSLATOR[temporal_res]}_name'])[0]}\n")
        file.write(f"MAX DATE: {np.sort(combined_gdf[f'{DATE_NAME_TRANSLATOR[temporal_res]}_name'])[-1]}\n")

        # total pct by counts
        file.write("DISTRIBUTION OF COUNTS, WHOLE DATASET: ")
        unique_vals, counts = np.unique(combined_gdf['counts'], return_counts=True)
        unique_vals = unique_vals.astype(int)
        counts = counts.astype(int)
        file.write(str({val: ct for val, ct in zip(unique_vals, counts)}))
        file.write("\n\n")

        # unique timesteps
        file.write('UNIQUE TIMESTEPS AND THEIR COUNTS:\n')
        
        for tstep in np.sort(combined_gdf[f"{DATE_NAME_TRANSLATOR[temporal_res]}_name"].unique()):
            filtered_gdf = combined_gdf[combined_gdf[f"{DATE_NAME_TRANSLATOR[temporal_res]}_name"] == tstep]
            unique_vals, counts = np.unique(filtered_gdf['counts'], return_counts=True)
            unique_vals = unique_vals.astype(int)
            counts = counts.astype(int)
            count_str = str({val: ct for val, ct in zip(unique_vals, counts)})
            file.write(f"{tstep}: {count_str}\n")
        
        # counts 

    if complete_idx_square:
        tstep_ids = combined_gdf.index.get_level_values(f'{DATE_NAME_TRANSLATOR[temporal_res]}_id').unique()
        geometries = combined_gdf.index.get_level_values('geometry').unique()
        full_index = pd.MultiIndex.from_product([tstep_ids, geometries], names=[f'{DATE_NAME_TRANSLATOR[temporal_res]}_id', 'geometry'])
        combined_gdf = combined_gdf.reindex(full_index)

    combined_gdf = combined_gdf.drop(columns=['geometry_col']).droplevel(1).reset_index()

    print('unique indicators', combined_gdf['month_indicator'].unique())
    return combined_gdf


DataSet = namedtuple('DataSplit', ['x', 'y', 'info'])

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

def make_outcome_history_feat_names(W, outcome_col='counts'):
    return ['prev_%s_%02dback' % (outcome_col, W - ww) for ww in range(W)]

def determine_season_year(date):
    return date.year if date.month >= 7 else date.year - 1

def clip_by_month(gps, timescale, first_month=11, last_month=4):
    """
    Given that gps data tracks birds all through the year, we need some way to filter the weeks that have no birds

    ARGUMENTS: 
        gps: gps dataframe we are going to cut from, by-season. 
        first_month, last_month: months that we clip out.
    RETURNS: dataframe with some all-zero weeks cut out, based on first and last month
    """
    gps = gps.sort_index()

    if timescale == 'seasonal':
        return gps, None
    
    cut_gps = gps[(gps['month'] >= first_month) | (gps['month'] <= last_month)]
    
    weekly_counts = gps[[f'{DATE_NAME_TRANSLATOR[timescale]}_name', 'counts']].groupby(gps.index.get_level_values(1)).agg({f'{DATE_NAME_TRANSLATOR[timescale]}_name': 'first', 'counts': 'sum'})        
    weekly_counts['start_date'] = pd.to_datetime(weekly_counts[f'{DATE_NAME_TRANSLATOR[timescale]}_name'].str.split('_to_').str[0])
    weekly_counts['season_year'] = weekly_counts['start_date'].apply(determine_season_year)
    weekly_counts = weekly_counts[weekly_counts[f'{DATE_NAME_TRANSLATOR[timescale]}_name'].isin(cut_gps[f'{DATE_NAME_TRANSLATOR[timescale]}_name'].unique())]
    return cut_gps, weekly_counts


def clip_by_sightings(gps, timescale, type_='', pct_thresh=0.05):
    """
    Given that gps data tracks birds all through the year, we need some way to filter the weeks that have no birds

    ARGUMENTS:
    df: gps dataframe we are going to cut from
    type_: choose from ['by_season', 'first_crane', 'percentile']:
        by_month: cut all months of data between pre-set months. Defaults are November (11) and April (4). 
        first_crane: Keep all rows between the first and last whooping crane seen in a given season, with exception for one whooping crane
            (there is a single crane documented in september 2012 with many zeros afterward)
        percentile: For each distribution of cranes per-day in each season, only keep rows between the first and last instance 
            higher than some pct_thresh% of the mean birds seen per-day in that season. Seasons here are determined by all rows between 
            the first and last whooping crane seen in a given year (from july-july).

    RETURNS: dataframe with some all-zero weeks cut out, depending on type_
    """
    gps.sort_index(inplace=True)

    if type_ not in ['first_crane', 'percentile']:
        raise ValueError('please read documentation for type_')
    
    if type_ == 'first_crane':
        
        # create concept of "seasons"
        weekly_counts = gps[[f'{DATE_NAME_TRANSLATOR[timescale]}_name', 'counts']].groupby(gps.index.get_level_values(1)).agg({f'{DATE_NAME_TRANSLATOR[timescale]}_name': 'first', 'counts': 'sum'})        
        weekly_counts['start_date'] = pd.to_datetime(weekly_counts[f'{DATE_NAME_TRANSLATOR[timescale]}_name'].str.split('_to_').str[0])
        weekly_counts['season_year'] = weekly_counts['start_date'].apply(determine_season_year)

        # group by season, take a forward and backward cumsum, and filter by where both are more than say, 5
        def cumsums(season_group):
            season_group['cumsum_forward'] = season_group['counts'].cumsum() #.cumsum()
            season_group['cumsum_backward'] = season_group['counts'][::-1].cumsum()[::-1] 
            return season_group[(season_group['cumsum_forward'] > 1) & (season_group['cumsum_backward'] > 1)][[f'{DATE_NAME_TRANSLATOR[timescale]}_name', 'season_year']] # .drop(['cumsum_forward', 'cumsum_backward'], axis=1)

        new_gps = weekly_counts.groupby('season_year').apply(cumsums)
        return new_gps
    
    # TODO implement
    elif type_ == 'percentile':
        pass
        # same process as above but find mean count for each season and use that as the threshold


def enum_geoid(all_df, timescale):
    """
    Given raw df with points in boxes, set up data for modeling
    """
    if not (f'{DATE_NAME_TRANSLATOR[timescale]}_name' in all_df.columns and 'lat' in all_df.columns and 'long' in all_df.columns):
        raise ValueError('need week_name, lat, and long in columns')

    all_df['geoid'] = all_df.apply(lambda row: (row['lat'], row['long']), axis=1)
    unique_geoids = {geoid: index for index, geoid in enumerate(all_df['geoid'].unique())}
    all_df['geoid'] = all_df['geoid'].map(unique_geoids)
    
    if timescale != 'seasonal':
        all_df['year'] = pd.to_numeric(all_df[f'{DATE_NAME_TRANSLATOR[timescale]}_name'].apply(lambda w: w.split('-')[0]))

        if 'monthly' not in timescale:
            all_df['month'] = pd.to_numeric(all_df[f'{DATE_NAME_TRANSLATOR[timescale]}_name'].apply(lambda w: w.split('-')[1])) 

    return all_df


def create_context_df(x_df, y_df, info_df,
        first_year, last_year,
        context_size, lag_tsteps,
        year_col='year', timestep_col='timestep', outcome_col='deaths', map_size='small', debug=False):
    
    """
    Create individual train, valid, test dfs
    """
    x_df.sort_index(inplace=True)
    y_df.sort_index(inplace=True)
    info_df.sort_index(inplace=True)

    new_col_names = make_outcome_history_feat_names(context_size, outcome_col=outcome_col) 
    assert last_year >= first_year

    xs = []
    ys = []
    infos = []

    for eval_year in range(first_year, last_year + 1):
        t_index = info_df[info_df[year_col] == eval_year].index
        timesteps_in_year = t_index.unique(level=timestep_col).values
        timesteps_in_year = np.sort(np.unique(timesteps_in_year))
        if debug == True: 
            print(f't index for {eval_year}', t_index)
            print(f'tsteps in year for {eval_year}', timesteps_in_year)

        for tt, tstep in enumerate(timesteps_in_year):
            # Make per-tstep dataframes
            x_tt_df = x_df.loc[idx[:, tstep], :].copy()
            x_tt_df['timestep_feat'] = tstep
            y_tt_df = y_df.loc[idx[:, tstep], :].copy()
            full_context_size = len(y_tt_df) * context_size
            info_tt_df = info_df.loc[idx[:, tstep], :].copy()
            xhist_N = y_df.loc[idx[:, tstep-(context_size+lag_tsteps-1):(tstep-lag_tsteps)], outcome_col].values.copy()
            if len(xhist_N) < full_context_size:  # check if we have achieved the amount of context we need
                if debug: print(f'cannot use {tstep}')
                continue
            else:
                if debug: print(f'usable tstep {tstep}')

            x_hist_num_obs = xhist_N.shape[0]
            obs_per_context = x_hist_num_obs // context_size
            xhist_context = xhist_N.reshape((obs_per_context, context_size))

            for ctxt in range(context_size):
                x_tt_df[new_col_names[ctxt]] = xhist_context[:, ctxt]

            xs.append(x_tt_df)
            ys.append(y_tt_df)
            infos.append(info_tt_df)

    if len(xs) == 0:
        raise ValueError('dataset passed in to create_context_df does not have enough timesteps to adequately provide context')

    return DataSet(pd.concat(xs), pd.concat(ys), pd.concat(infos))


def create_task(
    gps,
    study='gps',
    start_year = 2009,
    end_year = 2016,
    timescale = 'seasonal',
    box_length_m=500,
    season_clip_method='first_crane',
    lag_tsteps = 1,
    context_size = 3,
    geography_col = 'geoid',
    outcome_col = 'counts',
    year_col='season_id_year',
    map_size='small'):

    """
    overall function that creates the task
    """ 
    # cut to while  

    timestep_col = f'{DATE_NAME_TRANSLATOR[timescale]}_id'

    # cut to after tstep 137 (1994 fall)
    gps = enum_geoid(gps, timescale)

    if study == 'gps' and not (timescale == '2monthly' or timescale == '3monthly'):
        tr_years = [2009, 2010, 2011, 2012, 2013],
        va_years = [2014]
        te_years = [2015, 2016]

    else:
        # set percent of the data to use for train valid test
        train_pct = 0.85
        valid_pct = 0.05
        test_pct = 0.1

        if timescale == '2monthly' or timescale == '3monthly':
            start_year = gps[f"{DATE_NAME_TRANSLATOR[timescale]}_id"].min()
            end_year = gps[f"{DATE_NAME_TRANSLATOR[timescale]}_id"].max()

        tr_years = np.arange(153, 168, 1)
        va_years = np.arange(168, 174, 1)
        te_years = np.arange(174, 180, 1)
        # tr_years = np.arange(start_year, start_year + (end_year - start_year) * train_pct, 1, dtype='int')
        # va_years = np.arange(start_year + (end_year - start_year) * train_pct + 1, start_year + (end_year - start_year) * (train_pct + valid_pct) + 1, 1, dtype='int')
        # te_years = np.arange(start_year + (end_year - start_year) * (train_pct + valid_pct) + 2, end_year + 1, 1, dtype='int')
        print(f"train years: {tr_years}")
        print(f"valid years: {va_years}")
        print(f"test years: {te_years}")
    
    # function args
    x_cols = [timestep_col, 'lat', 'long', 'month_indicator']
    y_cols = [outcome_col]
    info_cols = [year_col]
    tr_years = np.sort(np.unique(np.asarray(tr_years)))
    va_years = np.sort(np.unique(np.asarray(va_years)))
    te_years = np.sort(np.unique(np.asarray(te_years)))
    assert np.max(tr_years) < np.min(va_years)
    assert np.max(va_years) < np.min(te_years)

    # Create the multiindex, reinserting timestep as a col not just index
    print(gps.columns)
    print(gps.index)
    print(geography_col, timestep_col)
    gps = gps.astype({geography_col: np.int64, timestep_col: np.int64})
    gps = gps.set_index([geography_col, timestep_col])
    gps[timestep_col] = gps.index.get_level_values(timestep_col)
    # geoid_key_df = gps.droplevel(1, axis=0)[['lat', 'long']]
    # geoid_key_df = geoid_key_df.loc[~info_df.index.duplicated(keep='first')]

    if season_clip_method == 'by_season' and timescale != 'seasonal' and timescale != '2monthly' and timescale != '3monthly':
        gps, valid_tsteps = clip_by_month(gps, timescale=timescale, first_month=11, last_month=4)

    # clip weeks with zeros
    if timescale != 'seasonal' and timescale != '2monthly' and timescale != '3monthly':
        
        print(f'clipping zero weeks by {season_clip_method}')
        
        if season_clip_method == 'by_season':
            gps, valid_tsteps = clip_by_month(gps, timescale=timescale, first_month=11, last_month=4)
        else:
            valid_tsteps = clip_by_sightings(gps, timescale=timescale, type_=season_clip_method, pct_thresh=0.05) # first_month=11, last_month=4,
            print(valid_tsteps)
        
        gps = gps[gps[f'{DATE_NAME_TRANSLATOR[timescale]}_name'].isin(valid_tsteps[f'{DATE_NAME_TRANSLATOR[timescale]}_name'].unique())]
        # map season year names to tsteps
        week_to_season = {w: s for w, s in zip(valid_tsteps[f'{DATE_NAME_TRANSLATOR[timescale]}_name'], valid_tsteps['season_year'])}
        gps['season_id_year'] = gps[f'{DATE_NAME_TRANSLATOR[timescale]}_name'].map(week_to_season)

    else:
        gps['season_id_year'] = gps[f'{DATE_NAME_TRANSLATOR[timescale]}_id']

    # start x/y split
    x_df = gps[x_cols].copy()
    y_df = gps[y_cols].copy()
    info_df = gps[info_cols].copy()

    tr_tup = create_context_df(x_df, y_df, info_df,
        tr_years[0], tr_years[-1],
        context_size, lag_tsteps,
        year_col=year_col, timestep_col=timestep_col, outcome_col=outcome_col)
    
    va_tup = create_context_df(x_df, y_df, info_df,
        va_years[0], va_years[-1],
        context_size, lag_tsteps,
        year_col=year_col, timestep_col=timestep_col, outcome_col=outcome_col)

    te_tup = create_context_df(x_df, y_df, info_df,
        te_years[0], te_years[-1],
        context_size, lag_tsteps,
        year_col=year_col, timestep_col=timestep_col, outcome_col=outcome_col)

    if 'save_files_torch_exps' in os.getcwd():
        path = '../data_dir'
    elif 'prob_diff_topk' in os.getcwd():
        path = 'data_dir'
    elif 'code' in os.getcwd():
        path = 'prob_diff_topk/data_dir'
    else:
        path = 'code/prob_diff_topk/data_dir'
 
    if not os.path.exists(f'data_dir/{timescale}_ctxtSize{context_size}_{map_size}Map'):
        os.makedirs(f'data_dir/{timescale}_ctxtSize{context_size}_{map_size}Map')       

    xtrain = tr_tup.x.droplevel(f'{DATE_NAME_TRANSLATOR[timescale]}_id').rename(columns={f'{DATE_NAME_TRANSLATOR[timescale]}_id': 'timestep'})
    xval = va_tup.x.droplevel(f'{DATE_NAME_TRANSLATOR[timescale]}_id').rename(columns={f'{DATE_NAME_TRANSLATOR[timescale]}_id': 'timestep'})
    xtest = te_tup.x.droplevel(f'{DATE_NAME_TRANSLATOR[timescale]}_id').rename(columns={f'{DATE_NAME_TRANSLATOR[timescale]}_id': 'timestep'})

    xtrain.to_csv(f'data_dir/{timescale}_ctxtSize{context_size}_{map_size}Map/bird_train_x.csv')
    xval.to_csv(f'data_dir/{timescale}_ctxtSize{context_size}_{map_size}Map/bird_valid_x.csv')
    xtest.to_csv(f'data_dir/{timescale}_ctxtSize{context_size}_{map_size}Map/bird_test_x.csv')

    ytrain = tr_tup.y.rename_axis(index={f'{DATE_NAME_TRANSLATOR[timescale]}_id': 'timestep'})
    yval = va_tup.y.rename_axis(index={f'{DATE_NAME_TRANSLATOR[timescale]}_id': 'timestep'})
    ytest = te_tup.y.rename_axis(index={f'{DATE_NAME_TRANSLATOR[timescale]}_id': 'timestep'})

    ytrain.to_csv(f'data_dir/{timescale}_ctxtSize{context_size}_{map_size}Map/bird_train_y.csv')
    yval.to_csv(f'data_dir/{timescale}_ctxtSize{context_size}_{map_size}Map/bird_valid_y.csv')
    ytest.to_csv(f'data_dir/{timescale}_ctxtSize{context_size}_{map_size}Map/bird_test_y.csv')

    print('DONE')

    # if not os.path.exists(f'{path}/{study}/{timescale}_ctxtSize{context_size}_{map_size}Map'):
    #     os.makedirs(f'{path}/{study}/{timescale}_ctxtSize{context_size}_{map_size}Map')

    # xtrain.to_csv(f'{path}/{study}/{timescale}_ctxtSize{context_size}_{map_size}Map/bird_train_x.csv')
    # xval.to_csv(f'{path}/{study}/{timescale}_ctxtSize{context_size}_{map_size}Map/bird_valid_x.csv')
    # xtest.to_csv(f'{path}/{study}/{timescale}_ctxtSize{context_size}_{map_size}Map/bird_test_x.csv')

    # ytrain = tr_tup.y.rename_axis(index={f'{DATE_NAME_TRANSLATOR[timescale]}_id': 'timestep'})
    # yval = va_tup.y.rename_axis(index={f'{DATE_NAME_TRANSLATOR[timescale]}_id': 'timestep'})
    # ytest = te_tup.y.rename_axis(index={f'{DATE_NAME_TRANSLATOR[timescale]}_id': 'timestep'})

    # ytrain.to_csv(f'{path}/{study}/{timescale}_ctxtSize{context_size}_{map_size}Map/bird_train_y.csv')
    # yval.to_csv(f'{path}/{study}/{timescale}_ctxtSize{context_size}_{map_size}Map/bird_valid_y.csv')
    # ytest.to_csv(f'{path}/{study}/{timescale}_ctxtSize{context_size}_{map_size}Map/bird_test_y.csv')

