
import argparse


# parameters for data preporcessing
# the start chunck from the list of individuals in all dataset 
# the end chunck from the list of individuals in all dataset
# the name of dataset 

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--start_chunck', type=int, help='The start chunck from the list of individuals in all dataset')
    parser.add_argument('--end_chunck', type=int, help='The end chunck from the list of individuals in all dataset')
    parser.add_argument('--dataset_name_input', type=int, help='The name of the dataset')
    

    
    args = parser.parse_args()

    dataset_name = {0: "Claim", 1: "NACRS", 2: "DAD", 3 : "PIN", 4: "Form10"}
    # If dataset is Claim, sheet number is 1, if dataset is NACRS, sheet number is 2, if dataset is DAD, sheet number is 3
    sheet_num = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}
    
    if args.end_chunck==-1:
        args.end_chunck=None

    args.dataset_name = dataset_name[args.dataset_name_input]
    args.sheet_num = sheet_num[args.dataset_name_input]
    args.new_columns=None
    
    if args.dataset_name=="Claim":
        args.headers =[
                        'PHN_ENC', 'RCPT_AGE_SE_END_YRS', 'age','RCPT_GENDER_CODE','DOCTOR_CLASS', 'SECTOR_DOCCLASS',
                        'Provider_specialty','SE_START_DATE', 'SE_END_DATE', 'DELV_SITE_FUNCTR_TYPE_CODE', 'DELV_SITE_FUNCTR_CODE_CLS', 
                        'DELV_SITE_TYPE_CLS', 'HLTH_DX_ICD9X_CODE_1', 'HLTH_DX_ICD9X_CODE_2','HLTH_DX_ICD9X_CODE_3']
        
        args.simple_headers =[
                        'subject_id', 'subject_age_at_service', 'subject_age_index', 'subject_sex', 'physician_type','physician_sector', 
                        'physician_specialty', 'start_date','end_date', 'service_type','service_code', 
                        'service_place', 'icd_diag_1', 'icd_diag_2', 'icd_diag_3']
        
        args.date='start_date'
        
    elif args.dataset_name=="NACRS":
        args.headers =[
                    'PHN_ENC','GENDER','age','RCPT_ZONE','PROV_PHN',  'AGE_ADMIT','ADMITBYAMB','VISIT_DATE','VISIT_TIME','VISIT_LOS_MINUTES','VISIT_MODE',
                    'TRIAGE_DATE','TRIAGE_TIME','TRIAGECODE','PIA_DATE','PIA_TIME', 'CACS_CODE','CACS_RIW','MAC','DISP_DATE',
                    'DISP_TIME','DISPOSITION','ED_DEPT_DATE','ED_DEPT_TIME', 'ED_ER_MINUTES','ED_VISIT_INDICATOR', 'EIP_MINUTES',
                    'INST', 'INST_ZONE', 'INSTFROM', 'INSTTO',
                    'PROCCODE1','PROCCODE2', 'PROCCODE3', 'PROCCODE4', 'PROCCODE5', 'PROCCODE6', 'PROCCODE7', 'PROCCODE8', 'PROCCODE9', 'PROCCODE10',
                    'DXCODE1', 'DXCODE2', 'DXCODE3', 'DXCODE4', 'DXCODE5', 'DXCODE6', 'DXCODE7', 'DXCODE8', 'DXCODE9', 'DXCODE10',

                    #'ABSTRACT_TYPE',
                    #  'MIS_CODE',
                    ]
        args.simple_headers =[
                    'subject_id', 'subject_sex', 'subject_age_index', 'subject_residence_zone', 'subject_province', 'subject_age_at_admit',
                    'admitted_by_ambulance', 'start_date', 'visit_time', 'stay_minutes',
                    'visit_method', 'triage_date', 'triage_time', 'triage_code', 'phys_init_assess_date',
                    'phys_init_assess_time', 'cacs', 'riw', 'mac', 'disposition_date', 'disposition_time',
                    'discharge_disposition', 'ed_departure_date', 'ed_departure_time', 'stay_ed_er_minitues',
                    'ed_visit_type', 'stay_er_inpatient', 'institution_num', 'institution_zone', 'institution_from',
                    'institution_to', 'icd_proc_1', 'icd_proc_2', 'icd_proc_3', 'icd_proc_4',
                    'icd_proc_5', 'icd_proc_6', 'icd_proc_7', 'icd_proc_8', 'icd_proc_9',
                    'icd_proc_10', 'icd_diag_1', 'icd_diag_2', 'icd_diag_3', 'icd_diag_4', 'icd_diag_5',
                    'icd_diag_6', 'icd_diag_7', 'icd_diag_8', 'icd_diag_9', 'icd_diag_10'
                    ]
        
        args.date='start_date'
        args.new_columns = 'emr_visit_type'

    elif args.dataset_name=="DAD":
    
        args.headers =[
                    'PHN_ENC','GENDER','age','RCPT_ZONE','PROVPHN','PSYCH_STAT','AGE_ADMIT','ADMITBYAMB','ADMITCAT','ADMITDATE','ADMITTIME',
                    'READMIT','ENTRYCODE','DOCSVC1','ACUTE_DAYS','ALL_DAYS', 'DISDATE','DISTIME','AGE_DISCH','INST','INST_ZONE','INSTFROM',
                    'INSTTO','ERDEPTDATE','ERDEPTTIME','CMG','MCC','COMASCALE','COMORB_LVL','RIW','DISP','DEADOR','DEADSCU','CCU_HOURS',
                    'ICU_HOURS','SCU1','SCU2', 'SCU3','SCU4','SCU5','SCU6','SCUHOURS1','SCUHOURS2','SCUHOURS3','SCUHOURS4','SCUHOURS5',
                    'SCUHOURS6','DXCODE1','DXCODE2', 'DXCODE3', 'DXCODE4', 'DXCODE5', 'DXCODE6', 'DXCODE7', 'DXCODE8', 'DXCODE9',
                    'DXCODE10', 'DXCODE11', 'DXCODE12', 'DXCODE13', 'DXCODE14', 'DXCODE15', 'DXCODE16', 'DXCODE17', 'DXCODE18',
                    'DXCODE19', 'DXCODE20', 'DXCODE21', 'DXCODE22', 'DXCODE23', 'DXCODE24', 'DXCODE25', 'PROCCODE1', 'PROCCODE2',
                    'PROCCODE3', 'PROCCODE4', 'PROCCODE5', 'PROCCODE6', 'PROCCODE7', 'PROCCODE8', 'PROCCODE9', 'PROCCODE10', 
                    'PROCCODE11', 'PROCCODE12', 'PROCCODE13', 'PROCCODE14', 'PROCCODE15','PROCCODE16', 'PROCCODE17','PROCCODE18',
                    'PROCCODE19','PROCCODE20' ]
        
        args.simple_headers =[
                'subject_id', 'subject_sex', 'subject_age_index', 'subject_residence_zone', 'subject_province', 'subject_psyc_status',
                'subject_age_at_admit','admitted_by_ambulance', 'admit_category', 'start_date', 'admit_time', 
                'readmition_code','last_entry_code_prior_admit','doc_svc_1', 'stay_days', 'stay_total_days', 'discharge_date', 'discharge_time', 'subject_age_at_discharge',
                'institution_num', 'institution_zone', 'institution_from', 'institution_to', 'er_departure_date', 'er_departure_time',
                'cmg', 'mcc', 'coma_scale_code', 'comorbidity_level', 'riw', 'discharge_disposition', 'died_in_intervention',
                'died_in_scu', 'ccu_hours', 'icu_hours', 'scu_1', 'scu_2', 'scu_3', 'scu_4',
                'scu_5', 'scu_6', 'scu_hours_1', 'scu_hours_2', 'scu_hours_3', 'scu_hours_4',
                'scu_hours_5', 'scu_hours_6', 'icd_diag_1', 'icd_diag_2', 'icd_diag_3', 'icd_diag_4',
                'icd_diag_5', 'icd_diag_6', 'icd_diag_7', 'icd_diag_8', 'icd_diag_9', 'icd_diag_10',
                'icd_diag_11', 'icd_diag_12', 'icd_diag_13', 'icd_diag_14', 'icd_diag_15', 'icd_diag_16',
                'icd_diag_17', 'icd_diag_18', 'icd_diag_19', 'icd_diag_20', 'icd_diag_21', 'icd_diag_22',
                'icd_diag_23', 'icd_diag_24', 'icd_diag_25', 'icd_proc_1', 'icd_proc_2',
                'icd_proc_3', 'icd_proc_4', 'icd_proc_5', 'icd_proc_6', 'icd_proc_7',
                'icd_proc_8', 'icd_proc_9', 'icd_proc_10', 'icd_proc_11', 'icd_proc_12',
                'icd_proc_13', 'icd_proc_14', 'icd_proc_15', 'icd_proc_16', 'icd_proc_17',
                'icd_proc_18', 'icd_proc_19', 'icd_proc_20'
       ]
        args.new_columns = 'Hosp_visit'
        args.date='start_date'



    elif args.dataset_name=="PIN":
        args.headers =['PHN_ENC', 'age', 'RCPT_GENDER_CD','DRUG_DIN',
                    'DSPN_AMT_QTY',
                    'DSPN_AMT_UNT_MSR_CD',
                    'DSPN_DATE',
                    'DSPN_DAY_SUPPLY_QTY',
                    'DSPN_DAY_SUPPLY_UNT_MSR_CD',
                    'SUPP_DRUG_ATC_CODE']
        
        args.simple_headers =['subject_id','subject_age_index','subject_sex','drug_number',
                            'drug_disp_amount_quant',
                            'drug_disp_amount_unit_measure',
                            'start_date',
                            'drug_disp_day_supply_quant',
                            'drug_disp_day_supply_unit_measure',
                            'drup_anatomical_chemical_code']
        
        args.date='start_date'

    elif args.dataset_name=="Form10":
        
        args.headers =['PHN_ENC', 'AUTHORED_DT', 'POLICE_CHECKBOX_VALUE']
        
        args.simple_headers =['subject_id','start_date','text']
        
        args.date='start_date'
        
    # Define separate lists for the common columns, subject columns, date columns, AMH columns, commorbidities columns, outcomes columns, and time difference columns
    args.cols_common = ['subject_id', 'subject_sex', 'subject_age_index', 'start_date', 'end_date', 'visit', 
                'substance_claim', 'mood_claim', 'anxiety_claim', 'psychotic_claim', 'cognitive_claim', 'otherpsych_claim', 'selfharm_claim',
                "substance_ER","mood_ER","anxiety_ER","psychotic_ER","cognitive_ER","otherpsych_ER","selfharm_ER"
                ,"substance_hos","mood_hos","anxiety_hos","psychotic_hos","cognitive_hos","otherpsych_hos","selfharm_hos",
                "antidepressant_medication","antipsychotic_medication","benzo_medication","mood_medication","cognitive_medication",




                'EX_CHF', 'EX_Arrhy', 'EX_VD', 'EX_PCD', 'EX_PVD', 'EX_HPTN_UC', 'EX_HPTN_C', 'EX_Para',
                'Ex_OthND', 'Ex_COPD', 'Ex_Diab_UC', 'Ex_Diab_C', 'Ex_Hptothy', 'Ex_RF', 'Ex_LD', 
                'Ex_PUD_NB', 'Ex_HIV', 'Ex_Lymp', 'Ex_METS', 'Ex_Tumor', 'Ex_Rheum_A', 'Ex_Coag', 
                'Ex_Obesity', 'Ex_WL', 'Ex_Fluid', 'Ex_BLA', 'Ex_DA', 'Ex_Alcohol', 'Ex_Drug', 
                'Ex_Psycho', 'Ex_Dep', 'Ex_Stroke', 'Ex_Dyslipid', 'Ex_Sleep', 'Ex_IHD', 'EX_Fall',
                'EX_Urinary', 'EX_Visual', 'EX_Hearing', 'EX_Tobacco', 'EX_Delirium', 'Ex_MS', 'EX_parkinsons',
                'detox','rehabilitation','addictions_counselling','unspecified_counselling','mental_health_therapy','occupational_therapy','medication_therapy',
                'homeless', 'police_interaction',
                'substance_time_diff_days', 'mood_time_diff_days', 'anxiety_time_diff_days', 
                'psychotic_time_diff_days', 'cognitive_time_diff_days', 'otherpsych_time_diff_days',
                'Ex_MS_time_diff_days', 'Ex_Dyslipid_time_diff_days']

    # Define separate lists for the subject columns, date columns, AMH columns, commorbidities columns, outcomes columns, and time difference columns
    args.cols_subject = ['subject_id', 'subject_sex', 'subject_age_index']
    args.cols_dates = ['start_date', 'end_date', 'visit']
    
    args.cols_AMH = ['substance_claim', 'mood_claim', 'anxiety_claim', 'psychotic_claim', 'cognitive_claim', 'otherpsych_claim', 'selfharm_claim']
    args.cols_AMH_ER = ["substance_ER","mood_ER","anxiety_ER","psychotic_ER","cognitive_ER","otherpsych_ER","selfharm_ER"]
    args.cols_AMH_hos = ["substance_hos","mood_hos","anxiety_hos","psychotic_hos","cognitive_hos","otherpsych_hos","selfharm_hos"]
    args.cols_AMH_med = ["antidepressant_medication","antipsychotic_medication","benzo_medication","mood_medication","cognitive_medication"]
    args.cols_AMH = args.cols_AMH + args.cols_AMH_ER + args.cols_AMH_hos + args.cols_AMH_med
    args.cols_commorbidities = ['EX_CHF', 'EX_Arrhy', 'EX_VD', 'EX_PCD', 'EX_PVD', 'EX_HPTN_UC', 'EX_HPTN_C', 'EX_Para',
                    'Ex_OthND', 'Ex_COPD', 'Ex_Diab_UC', 'Ex_Diab_C', 'Ex_Hptothy', 'Ex_RF', 'Ex_LD', 
                    'Ex_PUD_NB', 'Ex_HIV', 'Ex_Lymp', 'Ex_METS', 'Ex_Tumor', 'Ex_Rheum_A', 'Ex_Coag', 
                    'Ex_Obesity', 'Ex_WL', 'Ex_Fluid', 'Ex_BLA', 'Ex_DA', 'Ex_Alcohol', 'Ex_Drug', 
                    'Ex_Psycho', 'Ex_Dep', 'Ex_Stroke', 'Ex_Dyslipid', 'Ex_Sleep', 'Ex_IHD', 'EX_Fall',
                    'EX_Urinary', 'EX_Visual', 'EX_Hearing', 'EX_Tobacco', 'EX_Delirium', 'Ex_MS', 'EX_parkinsons']
    
    args.cols_outcomes = ['homeless', 'police_interaction']

    args.cols_therepy = ['detox','rehabilitation','addictions_counselling','unspecified_counselling','mental_health_therapy','occupational_therapy','medication_therapy']

    args.cols_time_diff = ['substance_time_diff_days', 'mood_time_diff_days', 'anxiety_time_diff_days', 
                    'psychotic_time_diff_days', 'cognitive_time_diff_days', 'otherpsych_time_diff_days',
                    'Ex_MS_time_diff_days', 'Ex_Dyslipid_time_diff_days']


    args.subject_id = 'subject_id'
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    return args



# python preprocessing.py --start_chunck 0 --end_chunck 100 --dataset_name_input 0