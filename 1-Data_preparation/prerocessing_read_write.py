

from datetime import timedelta
import os
import numpy as np
from tqdm import tqdm
import pyreadstat
import dask.dataframe as dd
import pandas as pd


# In[18]:
def return_list_of_codes_vars(file_excel_amhcohort_dict,dataset_name, sheet_num):

    # sheet_names[1] : is the one we have the claims in it
    # sheet_names[2] : is the one we have the nacrs in it
    # parse the related sheet of the excel related to the dataset_name
    print("The dataset name is: {}".format(dataset_name))

    df_amhcohort_dict=file_excel_amhcohort_dict.parse(file_excel_amhcohort_dict.sheet_names[sheet_num], na_values='')
        # Extract the headers
    headers = df_amhcohort_dict.columns.tolist()
    print("The headers in the excel file are: {}".format(headers))
    # seperate each variables related to each dataset

    # recive the variables and codes related to the dataset_name 
    df_dict=df_amhcohort_dict[df_amhcohort_dict['codes'].isnull()==False][headers]#[['Variable', 'codes','condition_times','condition_times_apart' ,'condition_period','mental_illness' ]]

    # inside the codes dictionary remove the space and other unnecessary  characters for simplicty of the camparison in the codes
    df_dict['codes']=df_dict['codes'].apply(lambda x:str(x).strip())
    df_dict['codes']=df_dict['codes'].apply(lambda x:str(x).replace('\'','').lower().strip())
    df_dict['codes']=df_dict['codes'].apply(lambda x:str(x).replace('\"','').lower().strip())
    df_dict['codes']=df_dict['codes'].apply(lambda x:str(x).replace(' ','').lower().strip())

   
    #in the following we will use the claim_code to creat coloumns for each diagnosis

    # add time diff field for each variable
    if dataset_name=='Claim':
        df_dict['Variable_time_diff']=df_dict['Variable'].map(lambda x: x.strip()+'_time_diff_days')

    if dataset_name=='DAD':
        df_dict['condition_icd_1']=df_dict['condition_icd_1'].apply(lambda x:str(x).strip())
        df_dict['condition_icd_1']=df_dict['condition_icd_1'].apply(lambda x:str(x).replace('\'','').lower().strip())
        df_dict['condition_icd_1']=df_dict['condition_icd_1'].apply(lambda x:str(x).replace('\"','').lower().strip())
        df_dict['condition_icd_1']=df_dict['condition_icd_1'].apply(lambda x:str(x).replace(' ','').lower().strip())
        

    df_dict=df_dict.fillna(0)
    return(df_dict)

# In[19]:

def to_read_sas_files(dir_of_files, df_dict, group_keys, dataset_name, headers, simple_headers,date, subject_id, new_columns=None): 

    pandas_dfs = []
    print("The number of individuals in the {} database --> calling in to_read_sas_files: {}".format(dataset_name,len(group_keys)))
    # Loop over each file in the directory
    n=0
    for file_name in tqdm(os.listdir(dir_of_files)):
    
        if file_name.endswith('.sas7bdat'):
            # print("The file name is: {}".format(file_name))
            file_path = os.path.join(dir_of_files, file_name)
            
            # Read the sas7bdat file into a pandas DataFrame
            # pandas_df = pd.read_sas(file_path, format='sas7bdat') #, chunksize=10000))
            pandas_df,meta=pyreadstat.read_sas7bdat(file_path)
            

            pandas_df=pandas_df[headers]
            
            # Rename the name of the columns for the simplicity
            pandas_df.columns=simple_headers
            
            if new_columns:
                pandas_df[new_columns]=str('')
            
            ###########################################################################################################
            # Add the columns (vriable in the dataset_name dict) for the mental illness/ or dignosis  from the Dictionary for the dataset_name
            for _,row in df_dict.iterrows():
                pandas_df[row.Variable]=False
                # if we have count condition for one variable, That means there needs to check the time differences between each record
                # for that variable
                if 'condition_times' in row:
                    if row.condition_times>0:
                        pandas_df[row.Variable_time_diff]=timedelta(days=-1) 
            
            
            # Filter the DataFrame based on subject IDs
            # print(len(group_keys))
            # filtered_df = pandas_df[pandas_df[subject_id].progress_apply(lambda group: group in list(group_keys))]

            # Optimized filtering with a progress bar
            # Convert group_keys to a set for O(1) average time complexity lookups
            group_keys_set = set(group_keys)

            # Use vectorized operation 
            mask = pandas_df[subject_id].isin(group_keys_set)

            filtered_df = pandas_df[mask]#.progress_apply(lambda x: x)
    
            
          
            # print("The number of individuals in the file: {}".format(len(filtered_df.groupby(subject_id))))
            # filtered_df=filtered_df.fillna(0)

            # Convert the pandas DataFrame to a Dask DataFrame
            # dask_df = dd.from_pandas(filtered_df, npartitions=1)  # Adjust the number of partitions as needed
            
            # Append the Dask DataFrame to the list
            pandas_dfs.append(filtered_df)
            n+=1
        

    # Concatenate all the Dask DataFrames into a single Dask DataFrame
    
    final_pandas_df = pd.concat(pandas_dfs, ignore_index=True).reset_index(drop=True)
    final_pandas_df = final_pandas_df.sort_values(by=[subject_id, date])
    final_dask_df = dd.from_pandas(final_pandas_df, npartitions=n)
 
    dask_df_gby_sub_id = final_dask_df.groupby(subject_id)
    expected_columns=pandas_df.columns
    print("the expected columns are: {}".format(expected_columns))

    return dask_df_gby_sub_id, expected_columns




# In[20]:


# Extracting the mental and physical illnesses based on the ICD codes in the Dictionary 
# At this point that we have all of the records for each individual, from the Claim database
# We need to extract the mental and physicall illness base on the ICD codes as follows :
def to_return_list_of_ind(dir_of_files, dataset_name):

    # check if the file is already exist in the hardware
    if os.path.exists(dir_of_files+'list_sub_id.h5'):
        final_df = pd.read_hdf(dir_of_files+'list_sub_id.h5', key='list_sub_id')

    else:

        pandas_dfs=[]
        # Loop over each file in the directory
        for file_name in tqdm(os.listdir(dir_of_files)):
            if file_name.endswith('.sas7bdat'):

                file_path = os.path.join(dir_of_files, file_name)
                
                # Read the sas7bdat file into a pandas DataFrame
                # pandas_df = pd.read_sas(file_path, format='sas7bdat') #, chunksize=10000))
                pandas_df,meta=pyreadstat.read_sas7bdat(file_path)

                pandas_df=pandas_df[['PHN_ENC']]
                pandas_df=pandas_df[pandas_df['PHN_ENC'].str.fullmatch(r'\d+')]
                # pandas_df['PHN_ENC'] = pandas_df['PHN_ENC'].str.decode('utf-8')
                
                # Append the Dask DataFrame to the list
                pandas_dfs.append(pandas_df)
                

        # Concatenate all the Dask DataFrames into a single Dask DataFrame
        final_df = pd.concat(pandas_dfs)
        final_df=final_df.drop_duplicates()
        final_df=final_df.sort_values(by=['PHN_ENC'])
        final_df=final_df.reset_index()
        
        
        # save the list of individuals in a file in the hardware for the further processing
        final_df.to_hdf(dir_of_files + 'list_sub_id.h5', key='list_sub_id', format='table')

    print("The number of individuals in the {} database: {}".format(dataset_name,len(final_df)))
    return final_df['PHN_ENC']

# In[21]:

def correct_types_write(df,str_df,args):

    num_start=args.start_chunck
    num_end=args.end_chunck
    dataset_name=args.dataset_name

    try:
        # Check the number of index levels
        if df.index.nlevels > 1:
            # Drop all levels of the MultiIndex
            df = df.reset_index(drop=True)
        else:
            # Drop the first level of the index
            df = df.reset_index(level=0, drop=True)
        # # Resetting the MultiIndex to a simple integer-based index
        # df = df.reset_index(level=1, drop=True)
        # df = df.reset_index(level=0, drop=True)

        if dataset_name=="Claim":
            name = 'df_claim_with_AMH'
        if dataset_name=="NACRS":
            name = 'df_nacrs'
        if dataset_name=="DAD":
            name = 'df_dad_with_AMH'
        if dataset_name=="PIN":
            name = 'df_pin_with_AMH'
        if dataset_name=="Form10":
            name = 'df_form10'

        # map a function  to_rename_maptypes_ based on the dataset_name
        rename_maptypes = globals()[f'to_rename_maptypes_{dataset_name.lower()}']
      
        df_gby = df.groupby('subject_id')
        print("The total amount of individuals from {} is {}".format(name,len(df_gby)))
        if 'police_interaction' in df.columns:
            print("Police interaction based on {}: {} individuals".format(dataset_name,len(df[df['police_interaction']])))
        if 'homeless' in df.columns:
            print("Homelessness based on {}: {} individuals".format(dataset_name,len(df[df['homeless']])))

        # print("Past Homelessness based on claim: {} individuals".format(len(df_claim_with_MH[df_claim_with_MH['Homeless_Past']].groupby(level='subject_id'))))





        # Step 1: Identify columns with string[pyarrow] dtype
        pyarrow_string_columns = df.select_dtypes(include=["string[pyarrow]"]).columns
        
        
        # Step 2: Convert these columns to str dtype
        for col in pyarrow_string_columns:
            df[col] = df[col].astype(str)

        # call the rename_maptypes function to rename the columns and map the visit types
        df_renamed_mapped = rename_maptypes(df, args)
        # timedelta_columns = df_renamed_mapped.select_dtypes(include=["timedelta64[ns]"]).columns
        
        # for col in timedelta_columns:
        #     timedelta_columns[col] = pd.Timedelta(days=int(timedelta_columns[col].dt.days)) 
  

        # once have all of the records in the df_claim_new, we will write it down to the hardware for the further processing 
        df_renamed_mapped.to_hdf(str_df + f'{name}_{num_start}_{num_end}.h5', key=f'{name}')

    except Exception as e:
        print("An error occurred during data conversion and writing: {}".format(str(e)))
        num_indexes = df_renamed_mapped.shape[0]
        print(num_indexes)
        print(df_renamed_mapped.head(num_indexes))
        print(df_renamed_mapped.columns)




def add_columns_not_in_dataset(df, cols_common, cols_time_diff, col_list, database_name):

    for col in cols_time_diff:
        if col not in df.columns:
            df[col] = pd.Timedelta(days=-1)
        else:
            df[col] = df[col].map(lambda x: pd.Timedelta(days=int(x.days)))

      
    print(f"The columns that are not in the claim:")
    for col in cols_common:
        if col not in df.columns:
            col_list.append(col)
            print(col)


    # Create a DataFrame with the new columns set to False
    new_cols_df = pd.DataFrame(False, index=df.index, columns=col_list)

    # Concatenate the new columns to the original DataFrame
    df = pd.concat([df, new_cols_df], axis=1)

    # Create a copy to defragment the DataFrame
    df = df.copy()
    df_new = df[cols_common].copy()
    df_new.columns = cols_common
    df_new['database'] = database_name

    print("DONE")

    return df_new


# Function to rename columns and map visit types for claim dataset
def to_rename_maptypes_claim(df_claim, args ):
    cols_common=args.cols_common
    cols_time_diff=args.cols_time_diff
    col_list=[]
    specialty_mapping = {
        'General/Family Physicians (GP/FPs)': 'family_gp',
        'Psychiatry': 'psychiatry',
        'Neurology': 'neurology',
        'Internal Medicine': 'im'
    }
    df_claim['visit'] = df_claim['physician_specialty'].map(specialty_mapping).fillna('other')


    # add the columns that are not in the dataset
    df_claim_new= add_columns_not_in_dataset(df_claim, cols_common, cols_time_diff, col_list, database_name='claim')


    return df_claim_new

# Function to rename columns and map visit types for dad dataset
def to_rename_maptypes_dad(df_dad, args ):

    col_list=[]
    cols_common=args.cols_common
    cols_time_diff=args.cols_time_diff

    df_dad['visit'] = df_dad['Hosp_visit'].map(lambda x: 'hospitalized_NonMH' if x == 'NonMH' else 'hospitalized_MH')
    df_dad['end_date'] = df_dad['start_date']
    # add the columns that are not in the dataset
    df_dad_new= add_columns_not_in_dataset(df_dad, cols_common, cols_time_diff, col_list, database_name='dad')

    return df_dad_new

# Function to rename columns and map visit types for pin dataset
def to_rename_maptypes_pin( df_pin, args):
    cols_common = args.cols_common
    cols_time_diff=args.cols_time_diff
    col_list=[]


    # df_pin['start_date'] = df_pin['drug_disp_date']
    df_pin['end_date'] = df_pin['start_date']
    df_pin['visit'] = 'pharmacy'
    # df_pin['cognitive']=df_pin['cognitive_pin']


    df_pin_new= add_columns_not_in_dataset(df_pin, cols_common, cols_time_diff, col_list, database_name='pin')

    return df_pin_new

# Function to rename columns and map visit types for nacrs dataset
def to_rename_maptypes_nacrs(df_nacrs, args):
    col_list=[]
    cols_common=args.cols_common
    cols_time_diff=args.cols_time_diff

    visit_mapping = {
        'MH_non_elect': 'emr_MH_non_elect',
        'NonMH': 'emr_NonMH',
        'MH_elect': 'emr_MH_elect'
    }

    df_nacrs['visit'] = df_nacrs['emr_visit_type'].map(visit_mapping).fillna('emr_visit')
    df_nacrs['end_date'] = df_nacrs['start_date']


    df_nacrs_new= add_columns_not_in_dataset(df_nacrs, cols_common, cols_time_diff, col_list, database_name='nacrs')


    # df_nacrs_new['end_date'] = df_nacrs_new[['start_date', 'end_date']].apply(lambda x: x[0] if pd.isnull(x[1]) else x[1], axis=1)
    
    return df_nacrs_new

# Function to rename columns and map visit types for pin dataset
def to_rename_maptypes_form10( df_form10, args):
    cols_common= args.cols_common
    cols_time_diff=args.cols_time_diff
    col_list=[]


    # df_pin['start_date'] = df_pin['drug_disp_date']
    df_form10['end_date'] = df_form10['start_date']
    df_form10['visit'] = 'police'
    df_form10['police_interaction'] = True
    # df_form10['subject_sex'] = 
    # df_pin['cognitive']=df_pin['cognitive_pin']

    df_form10_new= add_columns_not_in_dataset(df_form10, cols_common, cols_time_diff, col_list, database_name='form10')
  
    return df_form10_new
