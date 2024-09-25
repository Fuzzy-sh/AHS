
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# # Physicain_Claim
# 
# #### preprocessing
# 
# 1- Read the claim files and concat them
# 
# 2- If there is no conditoin on the variable based on the claim dictionary, 
# 
#     - Add the varaible to the claim dataset
#     
#     - Check if the claim code for the variable (based on Dictionary) is in the icd_diag_1 , 2, 3
#     
#         * The value for related variable in the claim dataset is changed to True
#         
# 3- If there is count condition for the variable in cliam_dic (ex: 2 physicians claims at least 30 days apart within a 2 year period with one or more of the following codes)
# * this condition means the cumsum of the time_diff should be claculated and for each row, its following cumsum should be checked if it is between 30 and 730. 
# * So, a matrix with the lenght of all the rows with that specila diagnosis code has been built. at each column of that matrix the cumsum for each row has been calculated. then the codes will check if each cumsum is within 30 and 730 with respect to each column. 
# 
# ![image.png](attachment:image.png)
# 
# 

# In[16]:


# Ex:
# time_diff =  0, 10, 15, 30, 60, 700
# __________________________________________________________________________________________________________________
# cumsum_0= 0, 10, 25, 65,125,825 -->        |0_True,     |10_False,  |25_False,  |65_True,   |125_True,  |865_False
# cumsum_10= -10, 0, 15, 55,115,815 -->      |-10_False,  |0_True,    |15_False,  |55_True,   |115_True,  |855_False
# cumsum_15= -25, -15, 0, 40,100,800 -->     |-25_False,  |-15_False, |0_True,    |40_True,   |100_True,  |840_False
# cumsum_30= -65, -55, -40, 0,60,740 -->     |-65_False,  |-55_False, |-40_False, |0_True,    |60_True,   |800_False
# cumsum_60= -125, -115, -100, -60,0,680 --> |-125_False, |-115_False,|-100_False,|-60_False, |0_True,    |680_True
# cumsum_700= -825, -815, -800, -760,-700,0->|-825_False, |-815_False,|-800_False,|-760_False,|-700_False,|0_False
# ________________________________________________________________________________________________________________
# .........................................................................................................
# Then the logical_or for each column (i.e :cumsum_0) of the the matrix will show the dianosis for that start date.
# In to_fill_diagnosis fuction all of the diagnosis will be catched and the related variable will be filled in as true or false



# Wrapper function to include progress update
def process_group_with_progress(group, df_dict, func, pbar, args):
   
    result = process_group(group, df_dict, func, args)
    pbar.update(1)
    return result


# Define the function to process each partition
def process_group(group_df, df_dict, func, args):
   
    df_result = pd.DataFrame()
    tbl_new, any_amh_diag = func(group_df, df_dict, args)
    
    if any_amh_diag:
        df_result = pd.concat([df_result, tbl_new], ignore_index=True)
      
        return df_result
    else:   
        pass


def to_fill_diagnosis_claim(tbl,df_claims_dict, args):
    
    # create a dictionary for dataframes with ( to extract the rows with mental illness with condition and calculate the time_diff) 
    df={}  
    start_date_service=args.date
    # itterated through the claim dict
    flag_any_amh_diag=False
    for _,row in df_claims_dict.iterrows():
        
        # gain the claim_code for each variable from the dictionary
        claims_code=row.codes.split(',')
        claims_code=[code.lower().strip() for code in claims_code]
        # Func: check condition for the claim code. if the claim code in icd_diag is in this list
        check_icd=lambda x: True if len([code for code in claims_code if code in str(x).lower().strip()])>0  else False

        
        #Return True if time_diff between 30-730 and count >=2
        min_period=pd.Timedelta(days= int(row.condition_times_apart)) # 0 or 30
        max_period=pd.Timedelta(days=int(row.condition_period)) #730
        

        #Func: check the period time that should be in min and max period
        # check_min_max=lambda x: False if x>max_period else (False if x<min_period else True)

        # Revised lambda function using .any() to reduce to scalar boolean
        check_min_max = lambda x: np.logical_and(min_period <= x, x<= max_period)  
        # if x is not datetime, convert to days
        convert_to_days = lambda x: pd.Timedelta(days=int(x)) if not isinstance(x, pd.Timedelta) and pd.notna(x) else x
        # check_min_max = lambda x:  bool((min_period <= timedelta(x)) & (timedelta(x) <= max_period))
        # check_min_max = lambda x: ((min_period <= x) & (x <= max_period)).all() if not isinstance(((min_period <= x) & (x <= max_period)), bool) else bool(min_period <= x) & (x <= max_period)
        
        # ----------------------------------------------------------------------------------------------------
        
        # If we have any True variable (mentall illness) for each table, it will be added to this list. 
        list_var_new_check_in_range=[]
        
        # If there is a count condition: which is true for the mental_illness variables
        if(row.condition_times>1):
            
            # all of the records containing the code of the variables (mental_illness) will be extracted 
            # and put into a df_varaible for one individual 
            df[row.Variable]= tbl[(tbl.icd_diag_1.map(check_icd)) | (tbl.icd_diag_2.map(check_icd)) | (tbl.icd_diag_3.map(check_icd))].copy() 
            df[row.Variable] = df[row.Variable].loc[~df[row.Variable].index.duplicated(keep='first')]
            # we need to keep track of the lenght of df_variable 
            df_var_len=len(df[row.Variable])
           
            
            # if the number of records in the separated df_variable is greater than the count condition
            if df_var_len>=row.condition_times:
                # Func: calculate the diff_date for the start_date_service
                


                # Checking for duplicate indices
                # duplicates = df[row.Variable].index[df[row.Variable].index.duplicated()].tolist()
                # if len(duplicates):
                #     print(f"Duplicate indices: {duplicates}")

                # Removing duplicates by keeping the first occurrence
                
                # print(df[row.Variable]['start_date_service'])
                df[row.Variable][start_date_service]=pd.to_datetime(df[row.Variable][start_date_service]).sort_values(ascending=False) # from the latest date to the earliest date
                
                
                # print(df[row.Variable]['start_date_service'])
                df[row.Variable][row.Variable_time_diff]=df[row.Variable][start_date_service].diff(1).fillna(pd.Timedelta(days=0))

                # creating the matrix for cumsum 
                # ---------------------------------------------------------------------------------------------------------------------                
                # create a matrix with the size of lenght of the df_var to create cumsum for every diff_date
                # and its following rows
                for j in range(0, df_var_len):
                    
                    # for each diff_date we need a cumsum and a var_new to keep the true, and false for each 
                    # value of the cumsum 30<= x< = 730
                    var_new_cumsum='{:.0f}_{}_diff_cumsum'.format(j,row.Variable)
                    var_new_check_in_range='{:.0f}_{}_diff_'.format(j,row.Variable)
                    
                    # for the first cumsum, we will use the diff_date for the related variable 
                    if j==0:
                        
                        df[row.Variable][var_new_cumsum]=df[row.Variable][row.Variable_time_diff].cumsum()
                        # df[row.Variable][var_new_cumsum]=df[row.Variable][var_new_cumsum].apply(lambda x: (timedelta(days=x)))
                        # df[row.Variable][var_new_cumsum]=df[row.Variable][var_new_cumsum].fillna(timedelta(days=0))
                        # keep the coulun name of the first cumsum
                        var_first_cumsum=var_new_cumsum
                        
                    # for the second to the last clumn of the matrix of cumsume, the first cumsum will be used to calculate others
                    else:
                        
                        # for the second column of the matrix, we use the second number of the first cumsum 
                        # extract the number located in that cell by its index
                        num_idx=df[row.Variable][var_first_cumsum].index[j]
                        num_=df[row.Variable].loc[num_idx,var_first_cumsum]
                        
                        # second column cumsum = [first_column_cumsum]-the second number in the first cumsum
                        # df[row.Variable][var_new_cumsum]=[i-num_ for i in df[row.Variable][var_first_cumsum]]
                        df[row.Variable][var_new_cumsum]= df[row.Variable][var_first_cumsum].apply(lambda x: x-num_)
                        


                        # df[row.Variable][var_new_cumsum]=df[row.Variable][var_new_cumsum].apply(lambda x: x.days)
                        # df[row.Variable][var_new_cumsum]=df[row.Variable][var_new_cumsum].fillna(timedelta(days=0))
                    
                    
                    # after creating the cumsum column for each diff_date we check, every number that should be between min and max
                    # if it is in the range the related cell in the var_new_check_in_range will be true otherwise it is false

                    try:
                        df[row.Variable][var_new_check_in_range]=df[row.Variable][var_new_cumsum].map(convert_to_days)
                        
                    except KeyError:
                        print(df[row.Variable][var_new_cumsum].dtypes)

                    df[row.Variable][var_new_check_in_range]=df[row.Variable][var_new_cumsum].map(check_min_max)
                    # Optionally, to ensure the DataFrame is de-fragmented
                    df[row.Variable] = df[row.Variable].copy()
                   
                    try:
                        df_true_var=df[row.Variable][df[row.Variable][var_new_check_in_range]]

                    except KeyError:
                        print(df[row.Variable][var_new_check_in_range].dtypes)

                    # true_var_condition = df[row.Variable][var_new_check_in_range].apply(lambda x: x).any(axis=1)
                    # try:
                    #     # df_true_var=df[row.Variable].loc[df[row.Variable][var_new_check_in_range],:]
                    #     df_true_var=df[row.Variable][row.Variable].loc[var_new_check_in_range]
                    #     print(df[row.Variable][var_new_check_in_range])
                    # except:
                    #     df_true_var=pd.DataFrame()
                    #     print(f"KeyError:")
                    #     print(df[row.Variable][var_new_check_in_range])
                    #     print("end error")
                    # df_true_var1=df[row.Variable][df[row.Variable][var_new_check_in_range]]

                    # if df_true_var.equals(df_true_var1)== False:
                    #     print('The two dataframes are not equal')
                    #     print(df_true_var)

                    #     print(df_true_var1)

                    # except KeyError:
                    #     df_true_var=df[row.Variable].loc[df[row.Variable][var_new_check_in_range],:]
                        # print(f"KeyError: {var_new_check_in_range}")
                        # print(df[row.Variable][var_new_check_in_range])
                        # df_true_var=df[row.Variable][var_new_check_in_range]
                        # print(df_true_var)
                
                    # len of the rows with true valus in each column_check_range 
                    
                    
                    # if there is true values 
                    if df_true_var.empty == False:
                        # The first row with the value of the zero which is the earlier day of the 
                        # of the second diagnosis will become true as well 
                        idx_first=df[row.Variable].index[j]
                        df[row.Variable].loc[idx_first,var_new_check_in_range]=True
                        
                         # list of columns in the matrix the keep True or False for each diff_date
                        list_var_new_check_in_range.append(var_new_check_in_range)
                        # print(list_var_new_check_in_range)
                         
                #----------------------------------------------------------------------------------
                # after creating that matrix, we will go through the in range values and if any true value 
                # in all the rows of for the check in range varibles, the related variable will become true
                if(len(list_var_new_check_in_range)>=1):
                    try:
                        df[row.Variable][row.Variable]=df[row.Variable][list_var_new_check_in_range].apply(any,axis=1)
                    except:
                        df[row.Variable][row.Variable]=df[row.Variable][list_var_new_check_in_range].apply(lambda x : x).any(axis=1)
                        # print(df[row.Variable][row.Variable])
                    
                    if(row.mental_illness==1):
                        flag_any_amh_diag=True
            
            # In the table for the individual, the related vlaues will ba changed based on the codes and conditions
            tbl.loc[list(df[row.Variable].index),[row.Variable,row.Variable_time_diff]]=df[row.Variable][[row.Variable,row.Variable_time_diff]] 

        else:
            # If there is no condition for the variable, just search the code and convert the related value into true
            l=list(tbl[(tbl.icd_diag_1.map(check_icd)) | (tbl.icd_diag_2.map(check_icd)) | (tbl.icd_diag_3.map(check_icd))].index)
            tbl.loc[l,row.Variable]= True

   
    # only the ones with mental illness will be returned back 
    return (tbl, True) if flag_any_amh_diag else ('', False)




# in this fuction all of the diagnosis will be catched and the related variable will be filled in as true or false
def to_fill_diagnosis_nacrs(tbl,df_nacrs_dict, args):
    
    # create a dictionary for dataframes with ( to extract the rows with mental illness with condition and calculate the time_diff) 
    df={}  
    
    #######################################################################
    # MH is the Emergency to psychiatric facilities 
    # to check the visit and the type of the visit
    check_emr_visit_true=lambda x: x 
    check_emr_visit_false=lambda x: False if x else True
    check_emr_visit_type_0=lambda x: True if x==0 else False
    check_emr_visit_type_1=lambda x: True if x==1 else False
    #############################################################################
    
    # itterated through the NACRS dict
    for _,row in df_nacrs_dict.iterrows():
      # empty the list of idx of prevuios variable 
        list_idx_check_icd=[]
        
        # gain the NACRS_code for each variable from the dictionary
        nacrs_code=row.codes.split(',')
        nacrs_code=[code.lower().strip() for code in nacrs_code]

        # Func: check condition for the NACRS code. if the NACRS code in icd_diag is in this list
        check_icd=lambda x: True if len([code for code in nacrs_code if code in x.lower().strip()])>0  else False
        
        
      # ----------------------------------------------------------------------------------------------------

        # for each icd_1 to 10 check if the we can find codes related to the variable 
        for icd_num in range(1,11):
            list_idx_check_icd=list_idx_check_icd+list(tbl[row.Variable][(tbl['icd_diag_'+str(icd_num)].map(check_icd))].index)
        
        tbl.loc[list_idx_check_icd,row.Variable]= True
    

    # elective and not elective visit for mental illness
    list_mh_elective=list(tbl[(tbl.emr_visit.map(check_emr_visit_true)) & (tbl.ed_visit_type.map(check_emr_visit_type_0))].index)  
    list_mh_non_elective=list(tbl[(tbl.emr_visit.map(check_emr_visit_true)) & (tbl.ed_visit_type.map(check_emr_visit_type_1))].index)  
    list_none_mh=list(tbl[(tbl.emr_visit.map(check_emr_visit_false))].index)
    
    # set the related value for the elective or non elective emergency visit
    tbl.loc[list_none_mh,'emr_visit_type']='NonMH'
    tbl.loc[list_mh_elective,'emr_visit_type']='MH_elect'
    tbl.loc[list_mh_non_elective,'emr_visit_type']='MH_non_elect'


    return (tbl,True)



# in this fuction all of the diagnosis will be catched and the related variable will be filled in as true or false
def to_fill_diagnosis_dad(tbl,df_dad_dict, args):
    
    # create a dictionary for dataframes with ( to extract the rows with mental illness with condition and calculate the time_diff) 
   
    
    #######################################################################
    # MH is the hospitalized to psychiatric facilities 
    # func: To check the substr for the second letter of the institute number
    check_substr_inst=lambda x: True if x[1]=='0' else False
    
    # func: To check the entry code
    check_entrycode=lambda x: True if x.lower()=='e' else False
    
    # func: To check the doc_scv_1 ; for the non_mental_health it should not be equal to '00064'or '00065'
    check_not_docscv_1=lambda x: False if x=='00064' else True
    check_not_docscv_2=lambda x: False if x=='00065' else True
    
    # func: To check the doc_scv_1 ; for the mental_health it should be equal to '00064'or '00065'
    check_docscv_1=lambda x: True if x=='00064' else False
    check_docscv_2=lambda x: True if x=='00065' else False
    
    # check the instute no that if they are equal to these numbers, it is MH hospital visit
    check_inst_1=lambda x: True if x=='85669' else False
    check_inst_2=lambda x: True if x=='85137' else False
    check_inst_3=lambda x: True if x=='85138' else False
    check_inst_4=lambda x: True if x=='85572' else False
    check_inst_5=lambda x: True if x=='88668' else False
    
    #############################################################################
    # itterated through the DAD dict
    flag_any_amh_diag=False
    for _,row in df_dad_dict.iterrows():
        
      # empty the list of idx of prevuios variable 
        list_idx_check_icd=[]
        
        # gain the DAD_code for each variable from the dictionary
        dad_code=row.codes.split(',')
        dad_code=[code.lower().strip() for code in dad_code]
        
        # Func: check condition for the DAD code. if the DAD code in icd_diag is in this list
        check_icd=lambda x: True if len([code for code in dad_code if code in x.lower().strip()])>0  else False
    
        # Func: check condition for the DAD code. if the DAD code in icd_diag is not in this list
        check_icd_not_in=lambda x: False if len([code for code in dad_code if code.lower().strip() in x.lower().strip()])>0  else True
      
      # ----------------------------------------------------------------------------------------------------
        # If there is no condition for the variable 
        if (pd.isna(row.condition_icd_1)==False):
        # for each icd_1 to 25 check if the we can find codes related to the variable 
            for icd_num in range(1,26):
                list_idx_check_icd=list_idx_check_icd+list(tbl[row.Variable][(tbl['icd_diag_'+str(icd_num)].map(check_icd))].index)

        # if there is a condition    
        else:
            # copy the rows that condition is true for them 
            tbl_with_cond=tbl[tbl['icd_diag_1'].map(check_icd_not_in)].copy()
            # for each icd_1 to 25 check if the we can find codes related to the variable
            for icd_num in range(1,26):
                list_idx_check_icd=list_idx_check_icd+list(tbl_with_cond[row.Variable][(tbl['icd_diag_'+str(icd_num)].map(check_icd))].index)
        
        # IF the variable is mental illness flag_any_amh_diag becomes true to included only the individuals with mentall illness
        if (len(list_idx_check_icd)>0) and (row.mental_illness==1):
            flag_any_amh_diag=True
            
        # convert the variables to true for the related indexe in the list
        tbl.loc[list_idx_check_icd,row.Variable]= True
    
    # hospital visit with the reasons other than mental illness
    list_not_mh_HV=list(tbl[(tbl.institution_num.map(check_substr_inst)) & (tbl.last_entry_code_prior_admit.map(check_entrycode)) & (tbl.doc_svc_1.map(check_not_docscv_1)) & (tbl.doc_svc_1.map(check_not_docscv_2))].index)
    
    # hospital visit with the reason of mental illness
    list_mh_HV=list(tbl[((tbl.institution_num.map(check_substr_inst)) | (tbl.institution_num.map(check_inst_1)) | (tbl.institution_num.map(check_inst_2)) | (tbl.institution_num.map(check_inst_3)) | (tbl.institution_num.map(check_inst_4)) | (tbl.institution_num.map(check_inst_5))) & ((tbl.doc_svc_1.map(check_docscv_1)) |(tbl.doc_svc_1.map(check_docscv_2)))].index)
    
    # for the related indexes set the variables to nonMH or MH for the hospital visit
    tbl.loc[list_not_mh_HV,'Hosp_visit']='NonMH'
    tbl.loc[list_mh_HV,'Hosp_visit']='MH'
    
    # only the ones with mental illness will be returned back 
    return (tbl, True) if flag_any_amh_diag else ('', False)


# in this fuction all of the diagnosis will be catched and the related variable will be filled in as true or false
def to_fill_diagnosis_pin(tbl,df_pin_dict, args):
    
    # create a dictionary for dataframes with ( to extract the rows with mental illness with condition and calculate the time_diff) 
    df={}  
    
    #######################################################################

    for _,row in df_pin_dict.iterrows():
      # empty the list of idx of prevuios variable 
        list_idx_check_icd=[]
        
        # gain the PIN_code for each variable from the dictionary
        pin_code=row.codes.split(',')
        pin_code=[code.lower().strip() for code in pin_code]

        # Func: check condition for the PIN code. if the PIN code in icd_diag is in this list
        check_icd=lambda x: True if len([code for code in pin_code if code in x.lower().strip()])>0  else False
        
        
      # ----------------------------------------------------------------------------------------------------

#         l=list(tbl[(tbl.icd_diag_1.map(check_icd)) | (tbl.icd_diag_2.map(check_icd)) | (tbl.icd_diag_3.map(check_icd))].index)
#         tbl.loc[l,row.Variable]= True
        
#         if (pd.isna(row.condition_icd_1)==False):

        list_idx_check_icd=list(tbl[row.Variable][(tbl['drug_number'].map(check_icd))].index)
        tbl.loc[list_idx_check_icd,row.Variable]= True
        

 
    return (tbl,True) if len(list_idx_check_icd)>=1 else ('', False)



# in this fuction all of the diagnosis will be catched and the related variable will be filled in as true or false
def to_fill_diagnosis_form10(tbl,df_form10_dict, args):
    
    # create a dictionary for dataframes with ( to extract the rows with mental illness with condition and calculate the time_diff) 
    # df={}  
    
    #######################################################################

    # for _,row in df_form10_dict.iterrows():
#       # empty the list of idx of prevuios variable 
#         list_idx_check_icd=[]
        
#         # gain the PIN_code for each variable from the dictionary
#         pin_code=row.codes.split(',')
#         pin_code=[code.lower().strip() for code in pin_code]

#         # Func: check condition for the PIN code. if the PIN code in icd_diag is in this list
#         check_icd=lambda x: True if len([code for code in pin_code if code in x.lower().strip()])>0  else False
        
        
#       # ----------------------------------------------------------------------------------------------------

# #         l=list(tbl[(tbl.icd_diag_1.map(check_icd)) | (tbl.icd_diag_2.map(check_icd)) | (tbl.icd_diag_3.map(check_icd))].index)
# #         tbl.loc[l,row.Variable]= True
        
# #         if (pd.isna(row.condition_icd_1)==False):

#         list_idx_check_icd=list(tbl[row.Variable][(tbl['drug_number'].map(check_icd))].index)
#         tbl.loc[list_idx_check_icd,row.Variable]= True
        
        

 
    return (tbl,True) #if len(list_idx_check_icd)>=1 else ('', False)
        