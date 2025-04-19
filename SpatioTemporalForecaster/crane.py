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


def sort_date_to_range(input_date, temporal_res):
    """
    Determines the range in which a given date falls based on month-day pairs.
    
    Parameters:
    - input_date (datetime): The date to evaluate.
    
    Returns:
    - str: The range in the format "year-month-day_to_year-month-day".
    """
    # set possible dates for range
    yr = input_date.year
    curr_year = [datetime.strptime(f"{yr}-{md}", "%Y-%m-%d") for md in MONTH_PAIRS_TRANSLATOR[temporal_res]]
    last_year = [datetime.strptime(f"{yr - 1}-{md}", "%Y-%m-%d") for md in MONTH_PAIRS_TRANSLATOR[temporal_res]]
    next_year = [datetime.strptime(f"{yr + 1}-{md}", "%Y-%m-%d") for md in MONTH_PAIRS_TRANSLATOR[temporal_res]]

    # extract correct range
    all_dates = last_year + curr_year + next_year
    idx = np.searchsorted(all_dates, input_date)
    correct_range = f"{all_dates[idx - 1].date()}_to_{all_dates[idx].date()}"

    return correct_range


def set_date_range_2_3mo(year, temporal_res):

    # Specify the year

    if temporal_res == '2monthly':
        # Create a date range for the specific dates
        dates = [
            pd.Timestamp(year, 10, 20),    # October 20
            pd.Timestamp(year, 12, 24),    # December 24
            pd.Timestamp(year + 1, 2, 27),  # Feb 28
            pd.Timestamp(year + 1, 4, 30)      # April 30
        ]
    elif temporal_res == '3monthly':
        # Create a date range for the specific dates
        dates = [
            pd.Timestamp(year, 10, 20),    # October 20
            pd.Timestamp(year + 1, 1, 31),    # December 31
            pd.Timestamp(year + 1, 4, 30)      # April 30
        ]

    # Convert to a Pandas DateTimeIndex
    return pd.DatetimeIndex(dates)

def determine_season_year(date):
    
    # return nan if between april 31 and oct 19
    if (5 <= date.month <= 9) or (date.month == 10 and date.day < 20):
        return np.nan

    return date.year if date.month >= 7 else date.year - 1

import pandas as pd

def date_range_gap(curr_drange, last_drange):
    """
    Given two date-range strings of the form:
      - "YYYY-02-28_to_YYYY-04-30" (spring; order 2)
      - "YYYY-12-25_to_YYYY-02-28" (winter; order 1, assigned to the end year)
      - "YYYY-10-20_to_YYYY-12-25" (fall; order 3)
    returns the number of “steps” between them.
    """
    def parse_range(s):
        start_str, end_str = s.split("_to_")
        sy, sm, sd = start_str.split("-")
        ey, em, ed = end_str.split("-")
        # For winter period, we assign the order to the ending year.
        if sm == "12" and sd == "25":
            order = 1
            year = int(ey)
        elif sm == "02" and sd == "28":
            order = 2
            year = int(sy)  # spring: both dates in the same year
        elif sm == "10" and sd == "20":
            order = 3
            year = int(sy)  # fall: both dates in the same year
        else:
            raise ValueError("Unexpected date range format: " + s)
        return year, order

    # Parse each range
    last_year, last_order = parse_range(last_drange)
    curr_year, curr_order = parse_range(curr_drange)

    # “Linearize” the three periods per year by assigning each an index.
    last_index = last_year * 3 + (last_order - 1)
    curr_index = curr_year * 3 + (curr_order - 1)
    return curr_index - last_index

def is_valid_bimonth_name(name):
    """
    Returns True if the bimonth name is one of the three allowed formats:
      - "YYYY-02-28_to_YYYY-04-30"
      - "YYYY-12-25_to_YYYY-02-28"
      - "YYYY-10-20_to_YYYY-12-25"
    Otherwise returns False.
    """

    try:

        parts = name.split("_to_")
        if len(parts) != 2:
            return False
        start, end = parts
        s_year, s_month, s_day = start.split("-")
        e_year, e_month, e_day = end.split("-")
        # Spring: 02-28 -> 04-30; same year.
        if s_month == "02" and s_day == "28" and e_month == "04" and e_day == "30" and s_year == e_year:
            return True
        # Winter: 12-25 -> 02-28; note: the years differ (the winter period “belongs” to the year of Feb 28).
        if s_month == "12" and s_day == "25" and e_month == "02" and e_day == "28":
            return True
        # Fall: 10-20 -> 12-25; same year.
        if s_month == "10" and s_day == "20" and e_month == "12" and e_day == "25" and s_year == e_year:
            return True
        return False


    except Exception:
        return False

def recalibrate_bimonth_ids(gdf):
    """
    Given a GeoDataFrame (or DataFrame) with columns:
      - "bimonth" (integer ID) and
      - "bimonth_name" (date range string),
    this function does three things:
      1. Drops any row whose bimonth_name is not valid.
      2. Checks the gap between successive rows in both the 'bimonth' and 'bimonth_name'
         columns.
      3. Re-assigns (re-indexes) the bimonth IDs so that for each row, its ID equals
         the previous row's ID plus the gap implied by the date range names.
         (For example, if the name gap is 2 but the IDs jump by 3, the new ID will be
          previous_ID + 2.)
    """
    # Step 1: Drop rows with invalid bimonth_name.
    invalid_indices = []
    for idx, row in gdf.iterrows():
        name = row['bimonth_name']
        if not is_valid_bimonth_name(name):
            invalid_indices.append(idx)
    if invalid_indices:
        print("Dropping rows with invalid bimonth_name at indices:", invalid_indices)
        gdf = gdf.drop(index=invalid_indices).reset_index(drop=True)

    # (Optional) Sort by bimonth if not already sorted.
    gdf = gdf.sort_values("bimonth").reset_index(drop=True)

    # Step 2 & 3: Walk through the rows and reassign bimonth IDs.
    new_ids = []
    for i, row in gdf.iterrows():
        if i == 0:
            # For the first row, we can either keep the original ID or start anew.
            new_id = row['bimonth']
            new_ids.append(new_id)
        else:
            prev_name = gdf.loc[i - 1, 'bimonth_name']
            curr_name = row['bimonth_name']
            try:
                expected_gap = date_range_gap(curr_name, prev_name)
            except ValueError as err:
                # If parsing fails for some reason, skip this row.
                print(f"Error parsing row {i}: {err}. Dropping row.")
                continue
            # Calculate new ID as previous new ID plus the expected gap.
            new_id = new_ids[i - 1] + expected_gap
            # (Optional) Report if the original gap did not match.
            original_gap = row['bimonth'] - gdf.loc[i - 1, 'bimonth']
            # if original_gap != expected_gap:
                # print(f"Row {i}: original ID gap ({original_gap}) does not match expected ({expected_gap}). Resetting ID.")
            new_ids.append(new_id)
    gdf['bimonth'] = new_ids
    return gdf


def reindex_consecutive(col):
    # Get the unique values in sorted order.
    sorted_unique = sorted(col.unique())
    # Create a mapping: original value -> consecutive integer starting at 1.
    mapping = {old_val: new_val for new_val, old_val in enumerate(sorted_unique, start=1)}
    # Replace each value in the column with its new consecutive number.
    return col.map(mapping)


def read_asurv(years_through_2011=10, temporal_res='weekly', context_size=5, map_size='small', keep_geometry_col=True):
    """
    Reads raw asurv data
    Assigns each observation into a temporal bucket based on temporal resolution
    """
    
    # read and turn into a geopandas dataframe
    print()
    df = pd.read_csv('../data/raw/aerial_surv/WHCR_Aerial_Observations_1950_2011.txt', encoding='latin1', sep='\t')
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.X, df.Y), crs='EPSG:26914')
    
    # cut years based on function parameter
    gdf = gdf[gdf['Year'].isin(gdf['Year'].unique()[-years_through_2011:])]

    # add time resolution
    gdf['date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    
    if temporal_res == 'seasonal':

        gdf['season'] = gdf['date'].apply(determine_season_year)
        gdf['season_name'] = gdf['season']

    elif temporal_res == '2monthly' or temporal_res == '3monthly':

        gdf['season'] = gdf['date'].apply(determine_season_year)
        gdf = gdf.dropna(subset='season')
        gdf['season'] = gdf['season'].astype('int')

        all_dates = pd.DatetimeIndex([])
        
        for szn in gdf['season'].unique():

            # set month ID
            curr_dates = set_date_range_2_3mo(year=szn, temporal_res=temporal_res)
            gdf.loc[gdf['season'] == szn, DATE_NAME_TRANSLATOR[temporal_res]] = np.searchsorted(curr_dates, gdf[gdf['season'] == szn]['date'])
            gdf = gdf.sort_values(by=['season', DATE_NAME_TRANSLATOR[temporal_res]]).reset_index(drop=True)
            gdf[DATE_NAME_TRANSLATOR[temporal_res]] = pd.factorize(list(zip(gdf['season'], gdf[DATE_NAME_TRANSLATOR[temporal_res]])))[0] + 1

            all_dates = np.concatenate((all_dates, curr_dates))

        gdf[f'{DATE_NAME_TRANSLATOR[temporal_res]}_name'] = gdf['date'].apply(lambda d: sort_date_to_range(d, temporal_res))
        
        # only keep dates that fall in the designated 2 month range. So, nothing in the summertime (april to october)
        def extract_mo_day(date_rnge):
            return date_rnge.split('_')[0].split('-')[1] + '-' + date_rnge.split('_')[0].split('-')[2]

        valid_idxs = gdf[f'{DATE_NAME_TRANSLATOR[temporal_res]}_name'].apply(lambda rnge: extract_mo_day(rnge) != '04-30')
        gdf = gdf[valid_idxs]

        gdf = recalibrate_bimonth_ids(gdf)
            
    else:
        
        all_dates = pd.date_range(start=gdf['date'].min() - DateOffset(days=DATE_OFFSET_TRANSLATOR[temporal_res]), end=gdf['date'].max() + DateOffset(days=DATE_OFFSET_TRANSLATOR[temporal_res]), freq=DATE_RANGE_TRANSLATOR[temporal_res])
        gdf[DATE_NAME_TRANSLATOR[temporal_res]] = np.searchsorted(all_dates, gdf['date'])  
        # add names for weeks for data clarity
        bin_names = {i + 1: f'{all_dates[i].date()}_to_{all_dates[i + 1].date()}' for i in range(len(all_dates) - 1)}

        gdf['season'] = gdf['date'].apply(determine_season_year)
        gdf = gdf.dropna(subset=['season'])

        gdf[f'{DATE_NAME_TRANSLATOR[temporal_res]}_name'] = gdf[DATE_NAME_TRANSLATOR[temporal_res]].map(bin_names)
        gdf[DATE_NAME_TRANSLATOR[temporal_res]] = reindex_consecutive(gdf[DATE_NAME_TRANSLATOR[temporal_res]])

    gdf['count'] = gdf['WHITE'].fillna(0) + gdf['JUVE'].fillna(0) + gdf['UNK'].fillna(0) 

    if keep_geometry_col:
        columns_of_interest = ['date', f"{DATE_NAME_TRANSLATOR[temporal_res]}", f"{DATE_NAME_TRANSLATOR[temporal_res]}_name", 'X', 'Y', 'season', 'count', 'geometry']
    else:
        columns_of_interest = ['date', f"{DATE_NAME_TRANSLATOR[temporal_res]}", f"{DATE_NAME_TRANSLATOR[temporal_res]}_name", 'X', 'Y', 'season', 'count']

    if not os.path.exists(f'data_dir/{temporal_res}_ctxtSize{context_size}_{map_size}Map'):
        os.makedirs(f'data_dir/{temporal_res}_ctxtSize{context_size}_{map_size}Map') 

    gdf.sort_values(by='date').to_csv(f'data_dir/{temporal_res}_ctxtSize{context_size}_{map_size}Map/intermed_df.csv')

    return gdf.sort_values(by='date')[columns_of_interest]


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


def points_to_boxes(gdf, temporal_res, box_length_m, keep_geometry_col, complete_idx_square, study='asurv', map_size='small', context_size=5, **kwargs):
    
    """
    Given gdf with x;y;count;tstep, assign counts to boxes 
    """

    if 'years_through_2011' in kwargs.keys():
        years_through_2011 = kwargs['years_through_2011']
    if 'years_cut_from_back' in kwargs.keys():
        years_cut_from_back = kwargs['years_cut_from_back']
    if 'tsteps_to_study' in kwargs.keys():
        tsteps_to_study = kwargs['tsteps_to_study']
    else:
        tsteps_to_study = None
    
    all_gdfs = []
    for tstep in np.sort(gdf[f'{DATE_NAME_TRANSLATOR[temporal_res]}_name'].unique()):
        
        filtered_gdf = gdf[gdf[f'{DATE_NAME_TRANSLATOR[temporal_res]}_name'] == tstep]
        # Read in bounding box from data folder
        with open("../data/raw/aerial_surv/boxes_total_bounds.txt", "r") as file:
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
        full_grid['season_indicator'] = SEASONAL_TRANSLATOR[datetime.strptime(tstep.split('_')[0], '%Y-%m-%d').month]

        full_grid['year'] = datetime.strptime(tstep.split('_')[0], '%Y-%m-%d').year
        
        # print(f"unique counts for {tstep}: {full_grid['counts'].unique()}")
        all_gdfs.append(full_grid)
    
    combined_gdf = pd.concat(all_gdfs)
    # print('shape of combined GDF', combined_gdf.shape[0])
    
    # create lat long columns and save to CSV
    combined_gdf.set_index([f'{DATE_NAME_TRANSLATOR[temporal_res]}_id', 'geometry'], drop=True, inplace=True)
    
    combined_gdf['geometry_col'] = combined_gdf.index.get_level_values(1)
    combined_gdf = combined_gdf.set_geometry('geometry_col')
    
    centers = combined_gdf.geometry.centroid
    centers_latlong = centers.to_crs('EPSG:4326')
    combined_gdf['lat'] = centers_latlong.y
    combined_gdf['long'] = centers_latlong.x

    if map_size != 'full':

        # print(combined_gdf)
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

        # print(combined_gdf.columns)
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

    if tsteps_to_study:
        min_tstep = combined_gdf.index.get_level_values(f'{DATE_NAME_TRANSLATOR[temporal_res]}_id').to_series().max() - tsteps_to_study
        combined_gdf = combined_gdf[combined_gdf.index.get_level_values(f'{DATE_NAME_TRANSLATOR[temporal_res]}_id') > min_tstep]

    combined_gdf = combined_gdf.drop(columns=['geometry_col']).droplevel(1).reset_index()
    combined_gdf.to_csv(f'data_dir/{temporal_res}_ctxtSize{context_size}_{map_size}Map/final_gdf.csv')
    return combined_gdf


DataSet = namedtuple('DataSplit', ['x', 'y', 'info'])

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
    study='asurv',
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
        train_pct = 0.8
        valid_pct = 0.1
        test_pct = 0.1

        if timescale == '2monthly' or timescale == '3monthly':
            start_year = gps[f"{DATE_NAME_TRANSLATOR[timescale]}_id"].min()
            end_year = gps[f"{DATE_NAME_TRANSLATOR[timescale]}_id"].max()

            tr_years = np.arange(153, 168, 1)
            va_years = np.arange(168, 174, 1)
            te_years = np.arange(174, 180, 1)

        else:
            pass
        # tr_years = np.arange(start_year, start_year + (end_year - start_year) * train_pct, 1, dtype='int')
        # va_years = np.arange(start_year + (end_year - start_year) * train_pct + 1, start_year + (end_year - start_year) * (train_pct + valid_pct) + 1, 1, dtype='int')
        # te_years = np.arange(start_year + (end_year - start_year) * (train_pct + valid_pct) + 2, end_year + 1, 1, dtype='int')
        print(f"train years: {tr_years}")
        print(f"valid years: {va_years}")
        print(f"test years: {te_years}")
    
    # function args
    x_cols = [timestep_col, 'lat', 'long', 'season_indicator']
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



def main(temporal_res: str, context_size=5, box_length_m=500, map_size='small', **kwargs):
    """
    Converts whooping crane raw data to dataset format
    """


    if 'tsteps_to_study' in kwargs.keys():
        tsteps_to_study = kwargs['tsteps_to_study'] # set how many tsteps will be in train-val-test
            # 15 train, 6 val, 6 test
    else:
        tsteps_to_study = 32

    complete_idx_square = True
    keep_geometry_col = False
    save_shp_folder = False

    years_through_2011 = 60
    gdf = read_asurv(years_through_2011, temporal_res=temporal_res, context_size=context_size, map_size=map_size)

    gdf = points_to_boxes(gdf, temporal_res=temporal_res, box_length_m=box_length_m, 
    keep_geometry_col=keep_geometry_col, complete_idx_square=complete_idx_square, years_through_2011=years_through_2011, 
    map_size=map_size, context_size=5, tsteps_to_study=tsteps_to_study)

    x = df_to_tensor(gdf, temporal_res=temporal_res)
    y = df_to_y_tensor(gdf, temporal_res=temporal_res)

    # make sure lookback data was accurately shifted back
    tracts = x.shape[1]
    assert tracts == y.shape[1]
    for i in range(tracts):
        # index 4 of dim 3 of X = target y at timestep t-1
        assert x[2, i, 4] == y[1, i]

    print(x, y)
    print('done')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--temporal_res', default='2monthly')
    parser.add_argument('--context_size', type=int, default=5)    
    parser.add_argument('--map_size', type=str, default='small')    
    parser.add_argument('--box_length_m', type=int, default=500)
    parser.add_argument('--tsteps_to_study', type=int, default=32)

    args = vars(parser.parse_args())
    print(args)

    main(**args)