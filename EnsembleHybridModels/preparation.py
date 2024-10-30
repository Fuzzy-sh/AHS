
'''
# The functions of the Preprocess class

#### 1- process_dataset: Rename the columns and create dummy variables 

#### 2- filter_records: The filtered group contains records up to the occurrence of the first 'homeless' or 'police_interaction' event, or the entire group if neither outcome is present.

#### 3-sort_and_group_with_filter: Sorting, and grouping based on subject_id and start date. Then apply filter_records for each individual.

#### 4-preprocess_events: Create a list of events for each date and put them in the events column. Then explode the 'events' column to create separate rows for each True value. Map each event to its corresponding ID using the 'event_dict' and create the 'events_id' column. It returns the preprocessed DataFrame with the 'events' column containing lists of events and the 'events_id' column contains unique integer IDs for each event.


#### 5- getRecordsUntilOutcome: results will be the filter dask data frame that calls functions 1,2, and 3.

#### 6- split_subject_ids: return the subjects list for train, test, train_w_val, and validation based on unique subject IDs gained from ddf data frame and 

####  7- split_datasets: Return data frames for training, testing, training with validation, and validation datasets based on subject lists created in 6.

'''


import dask.dataframe as dd
import numpy as np
from dask.diagnostics import ProgressBar
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os  # Standard library for interacting with the operating system


import ast
from sklearn.model_selection import train_test_split
from datetime import timedelta



def process_dataset(ddf):
    
    '''
    This function, 'process_dataset', performs essential data preprocessing steps on a Dask DataFrame. 
    It involves renaming columns for clarity and consistency, creating a new column based on existing ones, 
    categorizing a column to optimize memory and performance, generating dummy variables to convert categorical 
    data into binary columns, and selecting specific columns for the final dataset. Then convert the bool columns to the int type.
    Instead of sex {f and M}, we created dummy variables with 0 and 1 values for sex
    '''
    # Rename columns using a dictionary for clarity and consistency
    dataset = ddf.rename(columns={
           'EX_CHF':'CHF', 
           'EX_Arrhy':'Arrhy', 
           'EX_VD':'VD',
           'EX_PCD':'PCD', 
           'EX_PVD':'PVD', 
           'EX_HPTN_UC':'HPTN_UC', 
           'EX_HPTN_C':'HPTN_C', 
           'EX_Para':'Para', 
           'Ex_OthND':'OthND',
           'Ex_COPD':'COPD', 
           'Ex_Diab_UC':'Diab_UC', 
           'Ex_Diab_C':'Diab_C', 
           'Ex_Hptothy':'Hptothy', 
           'Ex_RF':'RF', 
           'Ex_LD':'LD',
           'Ex_PUD_NB':'PUD_NB', 
           'Ex_HIV':'HIV', 
           'Ex_Lymp':'Lymp', 
           'Ex_METS':'METS', 
           'Ex_Tumor':'Tumor', 
           'Ex_Rheum_A':'Rheum_A',
           'Ex_Coag':'Coag', 
           'Ex_Obesity':'Obesity', 
           'Ex_WL':'WL', 
           'Ex_Fluid':'Fluid', 
           'Ex_BLA':'BLA', 
           'Ex_DA':'DA',
           'Ex_Alcohol':'Alcohol', 
           'Ex_Drug':'Drug', 
           'Ex_Psycho':'Psycho', 
           'Ex_Dep':'Dep', 
           'Ex_Stroke':'Stroke',
           'Ex_Dyslipid':'Dyslipid', 
           'Ex_Sleep':'Sleep', 
           'Ex_IHD':'IHD', 
           'EX_Fall':'Fall', 
           'EX_Urinary':'Urinary',
           'EX_Visual':'Visual', 
           'EX_Hearing':'Hearing', 
           'EX_Tobacco':'Tobacco', 
           'EX_Delirium':'Delirium', 
           'Ex_MS':'MS',
           'EX_parkinsons':'parkinsons', 

    })

    # Create a new column 'homeless' based on 'homeless_past' and 'homeless_recent'
    dataset['homeless'] = dataset['homeless_past'] | dataset['homeless_recent']

    # Categorize the 'visit' column to optimize memory and performance
    dataset=dataset.categorize(columns=["visit"])

    # Create dummy variables for the 'visit' column to convert it into binary columns
    visit_df = dd.get_dummies(dataset['visit'], prefix='visit', dtype=bool)

    # Concatenate the dummy variables with the original DataFrame
    dataset = dd.concat([dataset, visit_df], axis=1,ignore_unknown_divisions=True)

    # Reorder and select desired columns for the final dataset
    columns = ['subject_id', 'sex', 'age', 'start_date',
               'visit_emr_MH_non_elect', 
               'visit_emr_NonMH', 
               'visit_emr_visit',
               'visit_family_gp', 
               'visit_hospitalized_NonMH', 
               'visit_im',
               'visit_neurology', 
               'visit_other', 
               'visit_psychiatry', 
               'visit_hosp_visit',
               'visit_hospitalized_MH', 
               'visit_pharmacy',

               'substance',
               'mood', 'anxiety', 'psychotic', 'cognitive', 'otherpsych',
               'selfharm', 


               'CHF', 'Arrhy', 'VD', 'PCD', 'PVD', 'HPTN_UC',
               'HPTN_C', 'Para', 'OthND', 'COPD', 'Diab_UC', 'Diab_C',
               'Hptothy', 'RF', 'LD', 'PUD_NB', 'HIV', 'Lymp', 'METS',
               'Tumor', 'Rheum_A', 'Coag', 'Obesity', 'WL', 'Fluid',
               'BLA', 'DA', 'Alcohol', 'Drug', 'Psycho', 'Dep', 'Stroke',
               'Dyslipid', 'Sleep', 'IHD', 'Fall', 'Urinary', 'Visual',
               'Hearing', 'Tobacco', 'Delirium', 'MS', 'parkinsons',


               'homeless', 'police_interaction']

    dataset = dataset[columns]
    
    print('# convert columns with boolean type to float 32')
    boolean_columns = [col for col in dataset.columns if dataset[col].dtype == 'bool']
    
    # Use 'astype' to convert the selected columns to float32
    dataset[boolean_columns] = dataset[boolean_columns].astype(np.float32)
    
    # Define a lambda function for min-max scaling
    # Compute the minimum and maximum and use them as constants
    min_age = 18
    max_age = 65
    
    min_max_scaling = lambda age: (age - min_age) / (max_age - min_age)
    
    print('# Apply min-max scaling to the age column using the lambda function')
    dataset['age'] = dataset['age'].map(min_max_scaling, meta=('age', 'float32'))
    
    # Define a lambda function to create dummy variables for 'sex'
    get_dummies_sex = lambda sex: 1 if sex == 'M' else 0
    
    # Apply the lambda function to create 'sex' dummy variables
    dataset['sex'] = dataset['sex'].map(get_dummies_sex, meta=('sex', 'float32'))

    # Remove duplicates
    # dataset = dataset.drop_duplicates()
    

    return(dataset)


##########################################################################################################

# Define a function to filter the records for each individual based on specific outcomes
# This function takes a group (a subset of the DataFrame grouped by subject_id) as input and returns the filtered group.

def filter_records(group):
    """
    Filter records for each individual based on the presence of 'homeless' or 'police_interaction' outcomes.

    Parameters:
        group (DataFrame): A subset of the original DataFrame grouped by subject_id.

    Returns:
        DataFrame: The filtered group containing records up to the occurrence of the first 'homeless' or 'police_interaction' event,
                   or the entire group if neither outcome is present.
    """

    # Check if the group contains any record with 'homeless' or 'police_interaction' being True
    has_homelessness = (group['homeless'] == True).any()
    has_police_interaction = (group['police_interaction'] == True).any()

    if has_homelessness and has_police_interaction:
        # If both outcomes are present, return the record related to the first outcome (homelessness or police interaction)
        # Get the index of the first occurrence of 'homeless' or 'police_interaction' based on 'start_date'
        outcome_index = group[(group['homeless'] == True) | (group['police_interaction'] == True)]['start_date'].idxmin()
        return group.loc[:outcome_index]

    elif has_homelessness:
        # If only homelessness outcome is present, return the records before the homelessness event
        # Get the index of the first occurrence of 'homeless' based on 'start_date'
        homelessness_index = group[group['homeless'] == True]['start_date'].idxmin()
        return group.loc[:homelessness_index]

    elif has_police_interaction:
        
        # If only police_interaction outcome is present, return the records before the police_interaction event
        # Get the index of the first occurrence of 'police_interaction' based on 'start_date'
        
        police_interaction_index = group[group['police_interaction'] == True]['start_date'].idxmin()
        
        return group.loc[:police_interaction_index]

    else:
        # If neither outcome is present, return the entire group
        return group

###########################################################################################################


def sort_and_group_with_filter(dataset):
    
    """
    
    Sorts the dataset first by 'subject_id' and then by 'start_date' within each 'subject_id' group.
    It then performs a groupby operation on 'subject_id' and applies a filtering function to each group.
    
    Parameters:
        dataset (DataFrame): The input DataFrame.
        
    Returns:
        DataFrame: The resulting  DataFrame after sorting, grouping, and filtering.
    """
    # Sort the dataset by 'subject_id' first
    dataset = dataset.sort_values(by='subject_id')#, kind='mergesort')

    # Sort the dataset by 'start_date' within each 'subject_id' group
    dataset = dataset.map_partitions(lambda df: df.sort_values(by='start_date'))#, kind='mergesort'))

    # Reset the index
    dataset = dataset.reset_index(drop=True)

    # Perform the groupby operation
    grouped_data = dataset.groupby('subject_id')


    # # Apply the filtering function to each group using map_partitions
    filtered = grouped_data.apply(filter_records)

    return filtered

###################################################################################################

def preprocess_events(filtered, start_events=4):
    
    """
    Preprocess the events in the DataFrame 'filtered' by converting True values in the columns 
    starting from 'start_events' into lists of event names for each row. It also creates an 'events_id' column
    containing unique integer IDs for each event.

    Parameters:
        filtered (pandas.DataFrame): The DataFrame containing the events to be preprocessed.
        start_events (int, optional): The index of the column from which events start in the DataFrame.
                                      Defaults to 4.
    Returns:
        pandas.DataFrame: The preprocessed DataFrame with the 'events' column containing lists of events
                          and the 'events_id' column contains unique integer IDs for each event.
    """

    # Select relevant columns starting from 'start_events'

    events_columns = filtered.iloc[:, start_events:].columns

    # Define a lambda function to create a list of strings with column names for each True value
    split_columns = lambda row: [col for col, value in row.items() if value]

    # Apply the lambda function to the relevant columns and create a new 'events' column
    filtered['events'] = filtered[events_columns].apply(split_columns, axis=1)

    # Explode the 'events' column to create separate rows for each True value
    filtered = filtered.explode('events')

    # Reset the index of the DataFrame
    filtered = filtered.reset_index(drop=True)

    # Get the unique list of events
    list_of_events = filtered['events'].unique()

    # Define a list of all possible events
    event_list = ['no event','visit_family_gp', 'visit_other', 'visit_emr_NonMH', 'anxiety', 'Dep', 'visit_emr_visit',
                  'substance', 'Tobacco', 'Hptothy', 'HPTN_UC', 'IHD', 'LD', 'visit_psychiatry', 'mood', 'Drug',
                  'visit_im', 'visit_neurology', 'Urinary', 'Fluid', 'visit_emr_MH_non_elect', 'otherpsych', 'Sleep',
                  'Alcohol', 'visit_hosp_visit', 'homeless', 'Hearing', 'Tumor', 'Dyslipid', 'COPD', 'Fall',
                  'psychotic', 'Psycho', 'RF', 'Rheum_A', 'visit_hospitalized_NonMH', 'Diab_C', 'Delirium', 'PCD',
                  'Diab_UC', 'Stroke', 'visit_hospitalized_MH', 'selfharm', 'Visual', 'Coag', 'OthND', 'VD', 'WL',
                  'Obesity', 'Lymp', 'DA', 'PVD', 'cognitive', 'MS', 'METS', 'CHF', 'Para', 'HPTN_C', 'HIV',
                  'police_interaction', 'parkinsons', 'Arrhy', 'visit_pharmacy', 'BLA', 'PUD_NB']

    # Create a dictionary to map each event to a unique integer index
    event_dict = {event: index for index, event in enumerate(event_list)}

    # Map each event to its corresponding ID using the 'event_dict' and create the 'events_id' column
    filtered['events_id'] = filtered['events'].map(event_dict)

    return filtered


def getRecordsUntilOutcome(ddf):
    """
    Return records up to the occurrence of the first outcome,
    or the entire group if no outcome is present.
    
    Parameters:
    - dask data frame: List of records to filter.
    
    Returns:
    - filtered data frame
    """

    # Process the dataset using the 'process_dataset' function
    print("Preparing the dataset using the 'process_dataset' function...")
    dataset = process_dataset(ddf)
    print('process_dataset finished successfully')
  
    
    # Get the number of records (rows) using the shape attribute
    num_records = dataset.shape[0]
    
    # Get the number of columns using the columns attribute
    num_columns = len(dataset.columns)
    
    print(f"Number of records (rows) of dataset: {num_records}")
    print(f"Number of columns of dataset: {num_columns}")
    
    # Print the number of partitions in the Dask DataFrame
    print('Number of Partitions:', dataset.npartitions)

    # Apply the filtering function to the dataset
    print("Applying the filtering function to the dataset...")
    filtered = sort_and_group_with_filter(dataset)


    return filtered



def split_subject_ids(ddf, id):
    
    print ('# Extract the Dask DataFrame as a Pandas DataFrame and reset the index')
    ddf_new = ddf.compute().reset_index(drop=True)
    
    print ("# Select the last row for each 'subject_id'")
    result_ddf = ddf_new.groupby(id).tail(1).reset_index(drop=True)
    
    print ('# Keep only the desired columns')
    result_ddf = result_ddf[[id, 'homeless', 'police_interaction']]
    
    print('# Split the data into training and testing sets based on subjects')
    test_set=result_ddf.groupby(['homeless', 'police_interaction'], group_keys=False).apply(lambda x: x.sample(frac=0.1))
    train_set=result_ddf[~result_ddf[id].isin(test_set[id])]
    
    
    test_set_ind_list=list(test_set[id])
    train_set_ind_list=list(train_set[id])
    
    
    print ("The number of individuals in the training set: {}".format(len(train_set_ind_list)))
    print ("The number of individuals in the test data set is: {}".format(len(test_set_ind_list)))
    
    return train_set_ind_list, test_set_ind_list


def split_datasets(ddf,ddf_name,id,train, test):

    print("Computing train, and test dataframes for : {}".format(ddf_name))

    train_data= ddf[ddf[id].isin(train)].reset_index(drop=True)
    print ("The number of records in the training set: {}".format(len(train_data)))
    print ("The number of individuals in the training set: {}".format(len(train_data[id].unique())))
    
    
    test_data = ddf[ddf[id].isin(test)].reset_index(drop=True)
    print ("The number of records in the test_data set: {}".format(len(test_data)))
    print ("The number of individuals in the test_data set: {}".format(len(test_data[id].unique())))
    
    
    return train_data,test_data


def add_events_to_folders(input_folder_path, output_folder_path):
    # Get the list of folders in the specified folder, excluding '.ipynb_checkpoints'
    folder_list = [folder for folder in os.listdir(input_folder_path) if os.path.isdir(os.path.join(input_folder_path, folder)) and folder != '.ipynb_checkpoints']

    for folder in folder_list:
        # Specify input and output folders
        input_folder = os.path.join(input_folder_path, folder)
        output_folder = os.path.join(output_folder_path, folder)

        print(f"Adding the events to the {folder} folder")

        # List all files in the input folder
        files = os.listdir(input_folder)

        # Iterate through each file
        for file in tqdm(files):
            # Check if the file is a Parquet file
            if file.endswith('.parquet'):
                
                # Construct the full path for both input and output
                input_path = os.path.join(input_folder, file)
                output_path = os.path.join(output_folder, file)

                # Read Parquet file into a Pandas DataFrame
                df = pd.read_parquet(input_path)

                # Perform your preprocessing here
                ddf_with_event = preprocess_events(df.copy())

                # For example, you can convert the Pandas DataFrame to a Dask DataFrame
                # ddf = dd.from_pandas(ddf_with_event, npartitions=1)

                # Write the processed data to another Parquet file with the same filename
                ddf_with_event.to_parquet(output_path)
