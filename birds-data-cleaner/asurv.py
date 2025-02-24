import pandas as pd
from pandas.tseries.offsets import DateOffset
from shapely.geometry import Polygon
import numpy as np
import geopandas as gpd
import os
import argparse
from datetime import datetime


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


def read_asurv(years_through_2011=10, temporal_res='weekly', context_size=5, map_size='small', keep_geometry_col=True):
    """
    Reads raw asurv data
    Assigns each observation into a temporal bucket based on temporal resolution
    """
    
    # read and turn into a geopandas dataframe
    df = pd.read_csv('asurv_1950_to_2011/WHCR_Aerial_Observations_1950_2011.txt', encoding='latin1', sep='\t')
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
            
    else:
        
        all_dates = pd.date_range(start=gdf['date'].min() - DateOffset(days=DATE_OFFSET_TRANSLATOR[temporal_res]), end=gdf['date'].max() + DateOffset(days=DATE_OFFSET_TRANSLATOR[temporal_res]), freq=DATE_RANGE_TRANSLATOR[temporal_res])
        gdf[DATE_NAME_TRANSLATOR[temporal_res]] = np.searchsorted(all_dates, gdf['date'])  
        # add names for weeks for data clarity
        bin_names = {i + 1: f'{all_dates[i].date()}_to_{all_dates[i + 1].date()}' for i in range(len(all_dates) - 1)}
        gdf[f'{DATE_NAME_TRANSLATOR[temporal_res]}_name'] = gdf[DATE_NAME_TRANSLATOR[temporal_res]].map(bin_names)

    gdf['count'] = gdf['WHITE'].fillna(0) + gdf['JUVE'].fillna(0) + gdf['UNK'].fillna(0) 

    if keep_geometry_col:
        columns_of_interest = ['date', f"{DATE_NAME_TRANSLATOR[temporal_res]}", f"{DATE_NAME_TRANSLATOR[temporal_res]}_name", 'X', 'Y', 'season', 'count', 'geometry']
    else:
        columns_of_interest = ['date', f"{DATE_NAME_TRANSLATOR[temporal_res]}", f"{DATE_NAME_TRANSLATOR[temporal_res]}_name", 'X', 'Y', 'season', 'count']

    if not os.path.exists(f'data_dir/{temporal_res}_ctxtSize{context_size}_{map_size}Map'):
        os.makedirs(f'data_dir/{temporal_res}_ctxtSize{context_size}_{map_size}Map') 

    gdf = recalibrate_bimonth_ids(gdf)

    gdf.to_csv(f'data_dir/{temporal_res}_ctxtSize{context_size}_{map_size}Map/intermed_df.csv')
    return gdf[columns_of_interest]