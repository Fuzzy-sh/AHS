import pandas as pd
import numpy as np
#Class Time-series multifeatured 
class TimeSeriesTensorGenerator:

    def __init__(self, observation_windows_start_column, observation_windows_end_column, prediction_windows_start_column, prediction_windows_end_column, demographic_start_column, demographic_end_column, prediction_lable1, prediction_lable2,prediction_value1, prediction_value2, onClinicDemCol):
        super(TimeSeriesTensorGenerator, self).__init__()
        
        # observation_windows; 4:-4 : for the tensor based 
        self.obs_win_start_col = observation_windows_start_column
        self.obs_win_end_col = observation_windows_end_column
        # prediction_windows; -4:-2 : for the tensor based 
        self.pred_win_start_col = prediction_windows_start_column
        self.pred_win_end_col = prediction_windows_end_column
        # demographic; 1:3 : for the tensor based
        self.demo_start_col = demographic_start_column
        self.demo_end_col = demographic_end_column
        # prediction_lable1: 'homeless'
        self.prediction_lable1=prediction_lable1
        # prediction_lable2: 'police_interaction'
        self.prediction_lable2= prediction_lable2
        # predction_value1: 1
        self.prediction_value1=prediction_value1 
        # predction_value2: 2
        self.prediction_value2=prediction_value2
        self.onClinicDemCol=onClinicDemCol
    
    def convert_timestamps_to_days(self, data_obs):
        # Extract the first column (timestamps)
        
        timestamps = data_obs['start_date'].copy()
        
        # Convert Timestamp objects and integers to datetime64
        timestamps_np = timestamps.map(lambda x: np.array(x, dtype='datetime64[s]') if isinstance(x, (int, np.integer)) else np.array(x, dtype='datetime64'))

        # Define a reference date
        reference_date = np.datetime64('2013-01-01')
    
        # Calculate the number of days since the reference date for each timestamp

        days_since_reference = (timestamps_np - reference_date) / np.timedelta64(1, 'D')

        # Replace the 'start_date' column with the calculated values
        data_obs.loc[:,'start_date'] = days_since_reference
    
        return data_obs

    def generate_tensors_by_windowlength(self, ind_db, observation_window, prediction_window, window_shift):
        
        """
        # Generate tensors based on the number of events
        Parameters:
        
            data (DataFrame): Pandas DataFrame containing "start_date" and "events_id" columns.
            prediction_window (int): The number of events in the prediction window.
            observation_window (int): The number of events in the observation window.
            window_shift (int): The number of events shifted for each chunk.
        
        Returns:
            time-series features in one matrix, its related outcome label, and demographics 
        """

        # # return the last date of record 
        # data_last_date=self['start_date'].iloc[-1]
        
        # # dataframe of the last record
        # data_last_db=self[self['start_date'] == data_last_date]
        
        # #dataframe of the all the data but the last record 
        # data_obs=self[self['start_date'] != data_last_date]
        # Get the last date of record using the index
        data_last_date = ind_db.index[-1]
        
        # DataFrame of the last record
        data_last_db = ind_db[ind_db.index == data_last_date]
        
        # DataFrame of all the data except the last record
        data_obs = ind_db[ind_db.index != data_last_date]

        # Calculate the range of start indices for observation windows
        len_obs=len(data_obs)
       
        # The observation data frame has fewer rows than the observation_window
        if len_obs < observation_window:
            
            # Append zero-filled rows directly to the original DataFrame
            data_obs = pd.concat([data_obs, pd.DataFrame(0, index=np.arange(observation_window - len_obs), columns=data_obs.columns)])
   
        else:
            # Calculate the number of chunks
            num_chunks = (len_obs + window_shift - 1) // window_shift
        
            # Calculate the total size after padding
            total_size = num_chunks * window_shift
        
            # Append zero-filled rows directly to the original DataFrame
            data_obs = pd.concat([data_obs, pd.DataFrame(0, index=np.arange(total_size - len_obs), columns=data_obs.columns)])

        len_obs=len(data_obs)
     
        # For padded the data observation, add the last records to the end as use it as the main dataset.
        db=pd.concat([data_obs,data_last_db])
        data_obs=self.convert_timestamps_to_days(data_obs)
        start_indices = range(0, len(data_obs) - observation_window + 1, window_shift)
        
        # Extract observation and prediction windows using DataFrame slicing

        # Create a list of selected data [4:-2]--> only columns of the time series features 
        # observation_windows = [data_obs.iloc[start_idx : start_idx + observation_window, self.obs_win_start_col:self.obs_win_end_col].values.astype(np.float32) for start_idx in start_indices]
        
        # gain the time_decay for each similar feature and each feature
        
        # Convert values to float32
        # observation_windows = [arr.astype(np.float32) for arr in observation_windows]
        
        # Create a list of outcome data [-2:]--> only homelessness and police interaction
        # prediction_windows = [db.iloc[start_idx + observation_window : start_idx + observation_window + prediction_window,self.pred_win_start_col:self.pred_win_end_col] for start_idx in start_indices]
        # Define the lambda function
        calculate_prediction_label = lambda pred_window: (pred_window[self.prediction_lable1] == 1).any() * self.prediction_value1 or (pred_window[self.prediction_lable2] == 1).any() * self.prediction_value2

        # Use the lambda function within the list comprehension
        combined_windows = [
            (
                data_obs.iloc[start_idx : start_idx + observation_window, self.obs_win_start_col:self.obs_win_end_col].values.astype(np.float32),
                calculate_prediction_label(db.iloc[start_idx + observation_window : start_idx + observation_window + prediction_window, self.pred_win_start_col:self.pred_win_end_col])
            )
            for start_idx in start_indices
        ]

        # combined_windows = [
        #     (
        #         data_obs.iloc[start_idx : start_idx + observation_window, self.obs_win_start_col:self.obs_win_end_col].values.astype(np.float32),
        #         (db.iloc[start_idx + observation_window : start_idx + observation_window + prediction_window, self.pred_win_start_col:self.pred_win_end_col][self.prediction_lable1] == 1).any() * self.prediction_value1 or (db.iloc[start_idx + observation_window : start_idx + observation_window + prediction_window, self.pred_win_start_col:self.pred_win_end_col][self.prediction_lable2] == 1).any() * self.prediction_value2
        #     )
        #     for start_idx in start_indices
        # ]
        observation_windows, prediction_label = zip(*combined_windows)
        
        # If there is any outcome in the prediction window, return its label 
        # prediction_label=[(prediction_window[self.prediction_lable1] == 1).any() * self.prediction_value1 or (prediction_window[self.prediction_lable2] == 1).any() * self.prediction_value2 for prediction_window in prediction_windows]

        # return the demographics, 'sex, age' as well.
        demographics=ind_db.iloc[:len(prediction_label),self.demo_start_col:self.demo_end_col].values
        
        # observation_windows: time series features in one matrix, its related outcome label, and demographics 
        return observation_windows, prediction_label, demographics

###########################################################################################################################
  
    def generate_oneWholeWindow_tensors_for_test_set(self, ind_db, obs_win_len, pred_win_len, window_shift=None):
        
        """
        This is for whole window " Accumulation window" to look for the next evnet using the whole window.

        Extract the observation window (all data except the last record)
        and the prediction window (the last record) from the dataset.
        
        Parameters:
            data (DataFrame): Pandas DataFrame containing the data.
        
        Returns:
            observation_window (DataFrame): The observation window.
            prediction_window (Series): The prediction window (last record).
        """

        # pred_win_len=1

        # Extract the observation window (all data except the last record)
        observation_window = [ind_db.iloc[:-pred_win_len,  self.obs_win_start_col:self.obs_win_end_col].values]
        observation_window=self.convert_timestamps_to_days(observation_window)
        # Separate the last record (prediction window)
        prediction_windows = ind_db.iloc[-pred_win_len:, self.pred_win_start_col:self.pred_win_end_col]
        
        calculate_prediction_label = lambda pred_window: (pred_window[self.prediction_lable1] == 1).any() * self.prediction_value1 or (pred_window[self.prediction_lable2] == 1).any() * self.prediction_value2

        prediction_label=[calculate_prediction_label(prediction_windows)]

        # If there is any outcome in the prediction window, return its label 
        # prediction_label=[(prediction_window[self.prediction_lable1] == 1).any() * self.prediction_value1 or (prediction_window[self.prediction_lable2] == 1).any() * self.prediction_lable2 for prediction_window in prediction_windows]

        # return the demographics, 'sex, age' as well.
        demographics=ind_db.iloc[:len(prediction_label),self.demo_start_col:self.demo_end_col].values
        
        return observation_window, prediction_label, demographics

    def generate_onClinicDem_tensor_for_test_set(self, ind_db, observation_window, prediction_window, window_shift=None):
        
        """
        This is for one the most last window of the individual in the test set, with the length of the observation and prediction window as it is in the training. 
        Extract the observation window (all data except the last record)
        and the prediction window (the last record) from the dataset.
        
        Parameters:
            data (DataFrame): Pandas DataFrame containing the data.
        
        Returns:
            observation_window (DataFrame): The observation window.
            prediction_window (Series): The prediction window (last record).
        """
        # if the type of trianing is tensor based: look for the features in the columns with values " substance usd disorder (7), ED Non-Mental health visit (6), unplnned mental heatlh visits(20), drug abuse (15)"
 
        # if the type of trianing is sequence based: look for the features in the sequences " substance usd disorder (7), ED Non-Mental health visit (6), unplnned mental heatlh visits(20), drug abuse (15)"
        # if self.model_name in ['GRU', 'GRU_with_timestamps']:
         
        # Get the boolean mask where any column in self.onClinicDemCol has a value greater than zero
        mask = (ind_db[self.onClinicDemCol] > 0).any(axis=1)

        # Use the boolean mask to filter the indices of ind_db
        onclinc_indices = np.where(mask)[0]
        # print(onclinc_indices)
        # print(ind_db)

        number_of_prediction_time=len(onclinc_indices)

        if number_of_prediction_time>0:

            # Get the last date of record using the index
            data_last_date = ind_db.index[-1]
            
            # DataFrame of the last record
            data_last_db = ind_db[ind_db.index == data_last_date]
            
            # DataFrame of all the data except the last record
            data_obs = ind_db[ind_db.index != data_last_date]
            
            # Calculate the range of start indices for observation windows
            len_obs=len(data_obs)
            # print(len_obs)
            # print(onclinc_indices[-1])
            # The observation data frame has fewer rows than the observation_window

            if len_obs < observation_window:
            
                # Append zero-filled rows directly to the original DataFrame
                data_obs = pd.concat([data_obs, pd.DataFrame(0, index=np.arange(observation_window - len_obs), columns=data_obs.columns)])

            if len_obs == onclinc_indices[-1]:
                data_obs = pd.concat([data_obs, pd.DataFrame(0, index=np.arange(1), columns=data_obs.columns)])

            if onclinc_indices[0]< observation_window:
                    
                    num_zero_before = observation_window - onclinc_indices[0]
            
                # Append zero-filled rows directly to the original DataFrame
                    data_obs = pd.concat([pd.DataFrame(0, index=np.arange(num_zero_before), columns=data_obs.columns), data_obs])
                
                    onclinc_indices = onclinc_indices + num_zero_before

            len_obs=len(data_obs)
    
            # For padded the data observation, add the last records to the end as use it as the main dataset.
            db=pd.concat([data_obs,data_last_db])
            data_obs=self.convert_timestamps_to_days(data_obs)
            calculate_prediction_label = lambda pred_window: (pred_window[self.prediction_lable1] == 1).any() * self.prediction_value1 or (pred_window[self.prediction_lable2] == 1).any() * self.prediction_value2

            # Use the lambda function within the list comprehension
            combined_windows = [
                (
                    data_obs.iloc[start_idx-observation_window+1:start_idx+1, self.obs_win_start_col:self.obs_win_end_col].values.astype(np.float32),
                    calculate_prediction_label(db.iloc[start_idx+1 : start_idx+ 1 + prediction_window, self.pred_win_start_col:self.pred_win_end_col]),
                    # db.iloc[:1,self.demo_start_col:self.demo_end_col].values.astype(np.float32)
                )
                for start_idx in onclinc_indices
            ]

            # combined_windows = [
            #     (
            #         data_obs.iloc[start_idx : start_idx + observation_window, self.obs_win_start_col:self.obs_win_end_col].values.astype(np.float32),
            #         (db.iloc[start_idx + observation_window : start_idx + observation_window + prediction_window, self.pred_win_start_col:self.pred_win_end_col][self.prediction_lable1] == 1).any() * self.prediction_value1 or (db.iloc[start_idx + observation_window : start_idx + observation_window + prediction_window, self.pred_win_start_col:self.pred_win_end_col][self.prediction_lable2] == 1).any() * self.prediction_value2
            #     )
            #     for start_idx in start_indices
            # ]

            observation_windows, prediction_label = zip(*combined_windows)

            # return the demographics, 'sex, age' as well.
            demographics= db.iloc[:len(prediction_label),self.demo_start_col:self.demo_end_col].values.astype(np.float32)
            
            # observation_windows: time series features in one matrix, its related outcome label, and demographics
        else:
            
            observation_windows, prediction_label, demographics = [],[],[]

        return observation_windows, prediction_label, demographics

    




    def generate_batchWindow_tensors_for_test_set(self,ind_db,  observation_window, prediction_window, window_shift):
        
        """

        batch windows for one individual. 
        The observation window 
        This is for one the most last window of the individual in the test set, with the length of the observation and prediction window as it is in the training. 
        Extract the observation window (all data except the last record)
        and the prediction window (the last record) from the dataset.
        
        Parameters:
            data (DataFrame): Pandas DataFrame containing the data.
        
        Returns:
            observation_window (DataFrame): The observation window.
            prediction_window (Series): The prediction window (last record).

        """

        # Get the last date of record using the index
        data_last_date = ind_db.index[-1]
        
        # DataFrame of the last record
        data_last_db = ind_db[ind_db.index == data_last_date]
        
        # DataFrame of all the data except the last record
        data_obs = ind_db[ind_db.index != data_last_date]

        # Calculate the range of start indices for observation windows
        len_obs=len(data_obs)

        # The observation data frame has fewer rows than the observation_window
        if len_obs < observation_window:
            
            # Append zero-filled rows directly to the original DataFrame
            data_obs = pd.concat([data_obs, pd.DataFrame(0, index=np.arange(observation_window - len_obs), columns=data_obs.columns)])
   
        else:
            # Calculate the number of chunks
            num_chunks = (len_obs + window_shift - 1) // window_shift
        
            # Calculate the total size after padding
            total_size = num_chunks * window_shift
        
            # Append zero-filled rows directly to the original DataFrame
            data_obs = pd.concat([data_obs, pd.DataFrame(0, index=np.arange(total_size - len_obs), columns=data_obs.columns)])

        len_obs=len(data_obs)
        
        # For padded the data observation, add the last records to the end as use it as the main dataset.
        db=pd.concat([data_obs,data_last_db])
        data_obs=self.convert_timestamps_to_days(data_obs)
        start_indices = range(0, len(data_obs) - observation_window + 1, window_shift)
        
        # Extract observation and prediction windows using DataFrame slicing

        # Create a list of selected data [3:-2]--> only columns of the time series features 
   
        # observation_windows = [data_obs.iloc[start_idx : start_idx + observation_window, self.obs_win_start_col:self.obs_win_end_col or None].values for start_idx in start_indices]

        
        # # gain the time_decay for each similar feature and each feature
        
        # # Convert values to float32
        # observation_windows = np.array([arr.astype(np.float32) for arr in observation_windows])
        
        # # Create a list of outcome data [-2:]--> only homelessness and police interaction
        # prediction_windows = [db.iloc[start_idx + observation_window : start_idx + observation_window + prediction_window,self.pred_win_start_col:self.pred_win_end_col or None] for start_idx in start_indices]
       
        # # If there is any outcome in the prediction window, return its label 
        # prediction_label=np.array([(prediction_window[self.prediction_lable1] == 1).any() * self.prediction_value1 or (prediction_window[self.prediction_lable2] == 1).any() * self.prediction_value2 for prediction_window in prediction_windows])

        calculate_prediction_label = lambda pred_window: (pred_window[self.prediction_lable1] == 1).any() * self.prediction_value1 or (pred_window[self.prediction_lable2] == 1).any() * self.prediction_value2

        # Use the lambda function within the list comprehension
        combined_windows = [
            (
                data_obs.iloc[start_idx : start_idx + observation_window, self.obs_win_start_col:self.obs_win_end_col].values.astype(np.float32),
                calculate_prediction_label(db.iloc[start_idx + observation_window : start_idx + observation_window + prediction_window, self.pred_win_start_col:self.pred_win_end_col])
            )
            for start_idx in start_indices
        ]

        # combined_windows = [
        #     (
        #         data_obs.iloc[start_idx : start_idx + observation_window, self.obs_win_start_col:self.obs_win_end_col].values.astype(np.float32),
        #         (db.iloc[start_idx + observation_window : start_idx + observation_window + prediction_window, self.pred_win_start_col:self.pred_win_end_col][self.prediction_lable1] == 1).any() * self.prediction_value1 or (db.iloc[start_idx + observation_window : start_idx + observation_window + prediction_window, self.pred_win_start_col:self.pred_win_end_col][self.prediction_lable2] == 1).any() * self.prediction_value2
        #     )
        #     for start_idx in start_indices
        # ]
        observation_windows, prediction_label = zip(*combined_windows)

        # return the demographics, 'sex, age' as well.
        demographics=ind_db.iloc[:len(prediction_label),self.demo_start_col:self.demo_end_col].values
     
        # observation_windows: time series features in one matrix, its related outcome label, and demographics 
        return observation_windows, prediction_label, demographics



