import pyreadstat
import os
import pandas as pd
from tqdm import tqdm
import gc
# Initialize tqdm to work with pandas
tqdm.pandas()

pd.options.mode.copy_on_write = True


class DataProcessor:
    def __init__(self):
        # Define constants
        self.cutoff_date = pd.Timestamp('2018-03-31')
        self.excluded_count = 0
        self.date_column = 'start_date'
        self.sub_id_column = 'subject_id'
        self.sub_sex_column = 'subject_sex'
        self.sub_age_column = 'subject_age_index'
        self.chunk_size = 50000
        self.data_folder = 'AHS_data/preprocessed_data/'
        self.raw_data_folder = 'AHS_data/raw_data/'

        # Define the columns for different aggregation methods
        self.first_columns = ['subject_sex', 'subject_age_index']
        self.time_diff_columns = ['substance_time_diff_days', 'mood_time_diff_days','anxiety_time_diff_days', 'psychotic_time_diff_days',
            'cognitive_time_diff_days', 'otherpsych_time_diff_days', 'Ex_MS_time_diff_days', 'Ex_Dyslipid_time_diff_days']
        self.first_columns =self.first_columns+self.time_diff_columns

        self.min_columns = ['start_date']
        self.max_columns = ['end_date']
        self.dommies_columns=['visit','database']



    def extract_data_claim_dad(self):
        print('Extracting the individuals data from Claim and DAD where AMH is available')
        folders = ['df_Claim', 'df_DAD']
        dfs_AMH = []
        for folder in folders:
            folder_path = os.path.join(self.data_folder, folder)
            file_list = [file for file in os.listdir(folder_path) if file.endswith('.h5')]
            print('Processing folder:', folder)
            for file in tqdm(file_list):
                file_path = os.path.join(folder_path, file)
                df = pd.read_hdf(file_path)
                dfs_AMH.append(df)

        print('Number of individuals with AMH in Claim and DAD:', pd.concat(dfs_AMH)[self.sub_id_column].nunique())
        return dfs_AMH

    def extract_data_pin(self, dfs_AMH):
        print('Extracting the individuals data from PIN where AMH is available')
        dfs_pin = []
        folder_path = os.path.join(self.data_folder, 'df_PIN')
        file_list = [file for file in os.listdir(folder_path) if file.endswith('.h5')]
        print('Processing folder:', folder_path)
        for file in tqdm(file_list):
            file_path = os.path.join(folder_path, file)
            df = pd.read_hdf(file_path)
            dfs_pin.append(df)

        combined_df_pin_AMH = pd.concat(dfs_pin).reset_index(drop=True)
        print('Number of individuals with AMH in PIN:', combined_df_pin_AMH[self.sub_id_column].nunique())
        ind_with_AMH_pin = combined_df_pin_AMH[combined_df_pin_AMH['cognitive_medication'] == True][self.sub_id_column].unique()
        filtered_df_AMH_pin = combined_df_pin_AMH[combined_df_pin_AMH[self.sub_id_column].isin(ind_with_AMH_pin)]
        dfs_AMH.append(filtered_df_AMH_pin)

        combined_df_AMH = pd.concat(dfs_AMH).reset_index(drop=True)
        print('Number of individuals with AMH gained from Claim, DAD, and PIN:', combined_df_AMH[self.sub_id_column].nunique())
        return combined_df_AMH

    def extract_data_registry(self, combined_df_AMH):
        print('Extracting the individuals data with AMH that have the insurance coverage data based on registry data')
        folder_path = os.path.join(self.raw_data_folder, 'Registry/')
        file_list = [file for file in os.listdir(folder_path) if file.endswith('.sas7bdat')]
        print('Processing folder:', folder_path)
        dfs_registry = []
        for file in file_list:
            file_path = os.path.join(folder_path, file)
            df, _ = pyreadstat.read_sas7bdat(file_path)
            dfs_registry.append(df)

        combined_df_registry = pd.concat(dfs_registry).reset_index(drop=True)
        individuals_with_cov = combined_df_registry[combined_df_registry.ACTIVE_COVERAGE == '1'].PHN_ENC.unique().tolist()
        individuals_with_AMH = combined_df_AMH[self.sub_id_column].unique()

        individuals_with_AMH_cov = list(set(individuals_with_AMH) & set(individuals_with_cov))
        print(f"Number of individuals with AMH: {len(individuals_with_AMH)}")
        print(f"Number of individuals with AMH with active coverage: {len(individuals_with_AMH_cov)}")
        return individuals_with_AMH_cov

    def extract_data_vital(self, individuals_with_AMH_cov):
        print('Extracting the individuals data with AMH and active insurance coverage who are alive between 2018 and 2020 based on Vital statistics data')
        folder_path = os.path.join(self.raw_data_folder, 'Vital_Statistics/')
        file_list = [file for file in os.listdir(folder_path) if file.endswith('.sas7bdat')]
        print('Processing folder:', folder_path)
        dfs_vital = []
        for file in file_list:
            file_path = os.path.join(folder_path, file)
            df, _ = pyreadstat.read_sas7bdat(file_path)
            dfs_vital.append(df)

        combined_df_vital = pd.concat(dfs_vital).reset_index(drop=True)
        individuals_dead = combined_df_vital[combined_df_vital.DETHDATE < '2018-03-31'].PHN_ENC.unique().tolist()
        individuals_with_AMH_cov_alive = list(set(individuals_with_AMH_cov) - set(individuals_dead))

        print(f"Number of individuals with AMH and insurance coverage: {len(individuals_with_AMH_cov)}")
        print(f"Number of individuals with AMH and insurance coverage who are alive as of '2018-03-31': {len(individuals_with_AMH_cov_alive)}")
        print('Saving the list of individuals with AMH and insurance coverage who are alive as of 2018-03-31')
        print("Number of individuals with AMH and insurance coverage who are dead as of 2018-03-31:", len(individuals_with_AMH_cov) - len(individuals_with_AMH_cov_alive))
        return individuals_with_AMH_cov_alive

    def extract_all_info(self, individuals_with_AMH_cov_alive):
        print('Extracting all the information from all the datasets for the individuals with AMH')
        folders = ['df_Claim', 'df_DAD', 'df_PIN', 'df_NACRS']
        dfs = []
        for folder in folders:
            folder_path = os.path.join(self.data_folder, folder)
            file_list = [file for file in os.listdir(folder_path) if file.endswith('.h5')]
            print('Processing folder:', folder)
            for file in tqdm(file_list):
                file_path = os.path.join(folder_path, file)
                df = pd.read_hdf(file_path)
                df=df[df[self.sub_id_column].isin(individuals_with_AMH_cov_alive)]
                df['start_date'] = pd.to_datetime(df['start_date'])
                df['end_date'] = pd.to_datetime(df['end_date'])
                dfs.append(df)

        print("Combining all the data ...")
        
        filtered_df_with_AMH_cov_alive = pd.concat(dfs).reset_index(drop=True)
        print(filtered_df_with_AMH_cov_alive.columns)
        # create a demographic dataframe
        demographic_df= filtered_df_with_AMH_cov_alive[[self.sub_id_column, self.sub_sex_column, self.sub_age_column]].drop_duplicates()

        # read the data from the form10
        folder_path = os.path.join(self.data_folder, 'df_Form10')
        # there is only one file in the folder so we can directly read it without looping
        file_list = [file for file in os.listdir(folder_path) if file.endswith('.h5')]
        file_path = os.path.join(folder_path, file_list[0])
        print('Processing folder:', folder_path)
        df_form10 = pd.read_hdf(file_path)

        # drop the subject_id column and age column from form10
        df_form10.drop(columns=[self.sub_sex_column, self.sub_age_column], inplace=True)
        # merge the demographic data with the form10 data on subject_id column based on inner join
        df_form10=pd.merge(demographic_df, df_form10, on=self.sub_id_column, how='inner')

        # rearrange the columns in the form10 based on the order of the columns in filtered_df_with_AMH_cov_alive
        df_form10=df_form10[filtered_df_with_AMH_cov_alive.columns]

        # merge the form10 data with the filtered_df_with_AMH_cov_alive
        filtered_df_with_AMH_cov_alive=pd.concat([filtered_df_with_AMH_cov_alive, df_form10], axis=0).reset_index(drop=True)


        print('Sorting the data based on subject_id and date')
        filtered_df_with_AMH_cov_alive.sort_values(by=[self.sub_id_column, self.date_column], ascending=[True, True], inplace=True)
        print("Changing the date format to datetime")
        # filtered_df_with_AMH_cov_alive[self.date_column] = filtered_df_with_AMH_cov_alive[self.date_column].map(lambda x: pd.to_datetime(x, errors='coerce'))
        # filtered_df_with_AMH_cov_alive['end_date'] = filtered_df_with_AMH_cov_alive['end_date'].map(lambda x: pd.to_datetime(x, errors='coerce'))

        return filtered_df_with_AMH_cov_alive

    def save_dfs_in_chunks(self, dfs, individuals_with_AMH_cov_alive, chunk_size, folder_path, outcome):
        num_chunks = len(individuals_with_AMH_cov_alive) // chunk_size + int(len(individuals_with_AMH_cov_alive) % chunk_size > 0)
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size

            chunk_subjects = individuals_with_AMH_cov_alive[start_idx:end_idx]
            chunk_df = dfs[dfs[self.sub_id_column].isin(chunk_subjects)]
            chunk_filename = os.path.join(folder_path, f'sub_{outcome}_{i+1}.h5')
            chunk_df.to_hdf(chunk_filename, key='df', mode='w')

            print(f"Saved {chunk_filename} with {len(chunk_subjects)} subjects.")

        print(f"All chunks have been saved for the {outcome} cohort.")


    # Define a function to exclude individuals based on the outcome that occurred before the cutoff date ( e.g., homeless, police_interaction)
    def exclude_ind_prior_outcome(self,dfs, outcome):

        # Step 1: Identify subject IDs to exclude based on the condition
        subject_ids_to_exclude = dfs.loc[(dfs[self.date_column] < self.cutoff_date) & (dfs[outcome]), self.sub_id_column].unique()

        # Step 2: Exclude those subject IDs from the DataFrame
        print(f"Total number of individuals : {dfs[self.sub_id_column].nunique()}")
        print(f"Total number of individuals  with {outcome} outcome: {len(dfs[dfs[outcome]][self.sub_id_column].unique())}")

        filtered_df = dfs[~dfs[self.sub_id_column].isin(subject_ids_to_exclude)]
        print(f"Number of individuals excluded with {outcome} outcome happened  before {self.cutoff_date}: {len(subject_ids_to_exclude)}")
        print(f"Number of individuals remaining: {filtered_df[self.sub_id_column].nunique()}")
        print(f"Number of individuals with first {outcome} happened during the following uo date: {len(filtered_df[filtered_df[outcome]][self.sub_id_column].unique())}")

        return filtered_df
    def fix_incom_values(self,df, column_name):
        sub_id_column='subject_id'
        # Create a mask for rows where the column has incorrect values (0 or False)
        mask_false_or_zero = (df[column_name] == 0) | (df[column_name] == False)

        # Get indices of the individuals with False or 0 values
        ind_with_false_val = df[mask_false_or_zero][sub_id_column].unique()
        print(f"Number of individuals with False or 0 value in {column_name} column: {len(ind_with_false_val)}")

        # Loop over unique individuals to correct values
        for ind in tqdm(ind_with_false_val):
            # Select rows for the current individual
            tbl = df[df[sub_id_column] == ind]
            
            # Get the correct value (first non-false/non-zero value in column)
            correct_val = tbl[(tbl[column_name] != False) & (tbl[column_name] != 0)][column_name].values[0]

            # Apply the correction using boolean indexing in one step
            df.loc[(df[sub_id_column] == ind) & mask_false_or_zero, column_name] = correct_val

        return df


    def aggregated_features(self, df, drop_first=False):
        # One-hot encode the specified columns
        if self.dommies_columns:
            dfs_encoded = pd.get_dummies(df, columns=self.dommies_columns, drop_first=drop_first)
        else:
            dfs_encoded = df.copy()
        # Create the aggregation dictionary
        agg_dict = {}
        
        # Add first occurrence columns to the aggregation dictionary
        if self.first_columns:
            agg_dict.update({col: lambda col: col.iloc[0] for col in self.first_columns})
        

        
        # Add min columns to the aggregation dictionary
        if self.min_columns:
            agg_dict.update({col: 'min' for col in self.min_columns})
        
        # Add max columns to the aggregation dictionary
        if self.max_columns:
            agg_dict.update({col: 'max' for col in self.max_columns})
        
        # Add boolean columns to the aggregation dictionary (sum their values)
        bool_columns = [col for col in dfs_encoded.columns if dfs_encoded[col].dtype == 'bool']
        agg_dict.update({col: 'sum' for col in bool_columns})
        
        # Perform the groupby and aggregation
        # grouped_df = dfs_encoded.groupby('subject_id').agg(agg_dict)
        # Group by 'subject_id' first
        grouped_by_subject = dfs_encoded.groupby('subject_id')

        # Define a function to process each group
        def process_group(group):
            # Ensure 'subject_id' is included in the aggregation result
            aggregated_group = group.agg(agg_dict)
            aggregated_group['subject_id'] = group['subject_id'].iloc[0]
            return aggregated_group
        
        # Apply the function to each subject group with progress tracking
        grouped_df = grouped_by_subject.progress_apply(process_group).reset_index(drop=True)
        
        
        return grouped_df
    
    def save_restults(self, restults, filename):
        restults.to_hdf(filename, key='df', mode='w')
        print(f"Results have been saved to {filename}")

    def drop_all_zero_columns(self, df):
        # Identify columns with all values zero
        zero_cols = df.columns[df.apply(pd.Series).eq(0).all()]
        print("Zero columns:")
        print(zero_cols)
        df.drop(zero_cols, axis=1, inplace=True)
        return df



def main():
    
    gc.collect()
    out_path='AHS_data/data/'
    agg_out_folder='AHS_data/cohorts/'

    homeless_filename = f'{agg_out_folder}homeless_cohort.h5'
    police_filename = f'{agg_out_folder}police_cohort.h5'
    # Save the result to a file
    
    alloucome='alloutcomes'
    homeless='homeless'
    police='police_interaction'
    processor = DataProcessor()

    dfs_AMH = processor.extract_data_claim_dad()
    combined_df_AMH = processor.extract_data_pin(dfs_AMH)
    individuals_with_AMH_cov = processor.extract_data_registry(combined_df_AMH)
    individuals_with_AMH_cov_alive = processor.extract_data_vital(individuals_with_AMH_cov)
    filtered_df_with_AMH_cov_alive = processor.extract_all_info(individuals_with_AMH_cov_alive)

    filtered_df_with_AMH_cov_alive=processor.fix_incom_values(filtered_df_with_AMH_cov_alive, 'subject_sex')
    filtered_df_with_AMH_cov_alive=processor.fix_incom_values(filtered_df_with_AMH_cov_alive, 'subject_age_index')
    filtered_df_with_AMH_cov_alive=processor.drop_all_zero_columns(filtered_df_with_AMH_cov_alive)
    processor.save_dfs_in_chunks(filtered_df_with_AMH_cov_alive, individuals_with_AMH_cov_alive, processor.chunk_size, out_path, outcome=alloucome)

    # Exclude individuals with homeless outcome before the cutoff date
    dfs_h=processor.exclude_ind_prior_outcome(filtered_df_with_AMH_cov_alive, outcome=homeless)
    processor.save_dfs_in_chunks(dfs_h, individuals_with_AMH_cov_alive, processor.chunk_size, out_path, outcome=homeless)
    agg_result_homeless = processor.aggregated_features(df=dfs_h , drop_first=False)
    processor.save_restults(agg_result_homeless, homeless_filename)      

    dfs_p=processor.exclude_ind_prior_outcome(filtered_df_with_AMH_cov_alive, outcome=police)
    processor.save_dfs_in_chunks(dfs_p, individuals_with_AMH_cov_alive, processor.chunk_size, out_path, outcome=police)
    agg_result_police = processor.aggregated_features(df=dfs_p , drop_first=False)
    processor.save_restults(agg_result_police, police_filename)

    print("All the functions have been executed successfully.")

if __name__ == '__main__':
    main()