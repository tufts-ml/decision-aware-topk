The purpose of this directory is to generate spatiotemporal maps of the aransas whooping crane population, grouped by boxes with preset sizes and preset temporal binning.

Sources: 
- asurv: https://iris.fws.gov/APPS/ServCat/DownloadFile/251560

Directories & Files:
- asurv.py: converts raw asurv data from .txt format to a workable dataset.
- clean_and_save.py: converts asurv raw dataframe to six workable dataframes for our top-k task

How to run program:
- First, add the data from the asurv download link (in "sources" section) to the folder asurv_1950_to_2011.
- clean_and_save.ipynb: run this script to generate data. the notebook will guide you through how to 
customize the data.
    - It is known that Python 3.11.7 will work, with the following packages: pandas, numpy, shapely, geopandas, os, ast, datetime, collections, sys, argparse

The notebook should create a data_dir folder with a subdirectory (with a name like 2monthly_ctxtSize5_smallMap to label timescale, context size, and map size) with the following files:
- bird_train_x.csv, bird_train_y.csv, bird_valid_x.csv, bird_valid_y.csv, bird_test_x.csv, bird_test_y.csv
    - files to run experiment
- EDA.txt: important information about the dataset that was created
- intermed_df.csv: point data of bird sightings before the data is grouped into boxes.

