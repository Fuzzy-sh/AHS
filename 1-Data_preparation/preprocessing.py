#!/usr/bin/python
# coding: utf-8



# improt the liberaries
import pyreadstat
import pandas as pd 

# from warnings import simplefilter
# simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

import numpy as np
from datetime import datetime, timedelta
import re
import os
from tqdm import tqdm
tqdm.pandas()  # This enables tqdm support for pandas
from tqdm.notebook import tqdm_notebook
tqdm_notebook.pandas()
# import dataframe_image as dfi
import time
# from alive_progress import config_handler
# import graphviz 
import h5py
from functools import partial

#recieving the parameters fromt the sh file 
import sys
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import prerocessing_read_write as pr_read_write
import processing_fill_diagnosis as pr_fill_diag
from utils import parse_args









# Main
####################################################################
# locations:


def main(args):


    str_data='AHS_data/'
    str_to_write='AHS_data/preprocessed_data/'
    str_to_read='AHS_data/raw_data/'

    # # All the codes are included in an dictionary excel file (amh_dict) and for each database there is separate sheet.
    # ### Description of data extraction:
    # # Identify people with addictions and mental health (AMH) using Claims, DAD and PIN databases and create a mental health and addictions study population.

    amh_dict_file = f"{str_data}amh_dict.xlsx"
    path_to_write=f"{str_to_write}df_{args.dataset_name}/" 
    path_to_read = f"{str_to_read}{args.dataset_name}/"


    # # read the file
    file_excel_amhcohort_dict=pd.ExcelFile(amh_dict_file)
    # # read the  dictionary and extract the variables and codes related to the dataset_name
    df_dict=pr_read_write.return_list_of_codes_vars(file_excel_amhcohort_dict , args.dataset_name, args.sheet_num)

    

    # merge all the  files and save the subject_id list in a file " ONLY ONCE to be saved and the rest to read the file and separate the individuals"
    # return all the list of individuals in dataset_name
    list_group_keys = pr_read_write.to_return_list_of_ind(path_to_read, args.dataset_name)
    print(f" Total Number of individuals in {args.dataset_name} : {len(list_group_keys)}")
    

    # the only the individuals in the chucnk of list will be read from the dataset_name files
    group_keys=list_group_keys[args.start_chunck: args.end_chunck]

    print(f" Total Number of individuals in this chunck from {args.dataset_name}: {len(group_keys)}")
    # call the function to read the sas files and extract the variables and codes related to the dataset_name
    df_gby_sub_id, expected_columns = pr_read_write.to_read_sas_files(path_to_read, df_dict , group_keys, args.dataset_name, args.headers, args.simple_headers, args.date, args.subject_id, args.new_columns)

    # create a meta data for the dask dataframe
    meta=pd.DataFrame(columns=expected_columns)

    # Initialize a counter for the total number of groups
    total_groups = len(group_keys)
    if args.dataset_name == "Claim":
        func = pr_fill_diag.to_fill_diagnosis_claim

    elif args.dataset_name == "NACRS":
        func = pr_fill_diag.to_fill_diagnosis_nacrs

    elif args.dataset_name == "DAD":
        func = pr_fill_diag.to_fill_diagnosis_dad

    elif args.dataset_name == "PIN":
        func = pr_fill_diag.to_fill_diagnosis_pin
    
    elif args.dataset_name == "Form10":
        func = pr_fill_diag.to_fill_diagnosis_form10
        
    #  Creating a dataframe to keep the individuals with mental illness
    with tqdm(total=total_groups, desc="Processing groups") as pbar:
  
        df_with_MH = df_gby_sub_id.apply(lambda group: pr_fill_diag.process_group_with_progress(group, df_dict, func, pbar, args), meta=meta).compute()


    # as discissed with the group, some individuals are in the Calgary cohort do not have Mentall illness
    # based on the condition stated in the Dictionary. However, they are included in the Calgary cohort. 
    # So after discussing, we need to exclude them from the cohort that we have. 
    # Python code t get difference between our individual with mental illness extracted from the dataset and 
    # the Calgary cohort database database.

    # do statisitcs and write the restuls 
    pr_read_write.correct_types_write(df_with_MH, path_to_write, args)

    print(f" The {args.dataset_name} for the chunck {args.start_chunck} to {args.end_chunck} is saved in the path {path_to_write}")

if __name__ == "__main__":

    args = parse_args()
    main(args)

# python preprocessing.py --start_chunck 0 --end_chunck 100 --dataset_name_input 0