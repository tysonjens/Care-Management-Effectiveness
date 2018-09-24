
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dateutil.parser import parse
from pandas.tseries.offsets import Day
from sklearn import linear_model
from sklearn.preprocessing import Imputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
# import warnings; warnings.simplefilter('ignore')
pd.set_option("display.max_columns", 100)
pd.set_option('display.max_rows', 500)
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Functions

# In[ ]:


## function that finds the most recent discharge before a program begins
def find_index_admit(programs, admissions):
    admissions = admissions.sort_values(by='discharge_date', ascending=False)
    index_dates = np.empty(programs.shape[0])
    index_dates[:] = np.nan
    index_dates = list(index_dates)
    for index, row in programs.iterrows():
        admit_pat = admissions[admissions['EMPI']==row['EMPI']]
        for index2, row2 in admit_pat.iterrows():
            if row2['discharge_date'] < row['prog_create_date']:
                index_dates[index] = row2['discharge_date']
                break
            else:
                continue
    return index_dates


# In[ ]:


## function that finds the first admission after a program begins
def find_first_admission_after_enroll(programs, admissions):
    admissions = admissions.sort_values(by='admit_date', ascending=True)
    first_admission = np.empty(programs.shape[0])
    first_admission[:] = np.nan
    first_admission = list(first_admission)
    for index, row in programs.iterrows():
        admit_pat = admissions[admissions['EMPI']==row['EMPI']]
        for index2, row2 in admit_pat.iterrows():
            if row2['admit_date'] > row['prog_create_date']:
                first_admission[index] = row2['discharge_date']
                break
            else:
                continue
    return first_admission


# In[ ]:


## function that finds the LACE score that occurred within 10 days (prior) to program start date
def find_lace_prior_to_enroll(programs, lace, window_size=40):
    lace = lace.sort_values(by='ASES_DT', ascending=False)
    lace_scores = np.empty(programs.shape[0])
    lace_scores[:] = np.nan
    lace_scores = list(lace_scores)
    for index, row in programs.iterrows():
        lace_pat = lace[lace['EMPI']==row['EMPI']]
        for index2, row2 in lace_pat.iterrows():
            if (row2['ASES_DT'] < row['prog_create_date']) & (row2['ASES_DT'] > (row['prog_create_date']-timedelta(days=window_size))):
                lace_scores[index] = row2['ASES_SCOR']
                break
            else:
                continue
    return lace_scores


# In[ ]:


def find_assessments_during_program(programs, assess, ases_nm='TOC Post Discharge Outreach'):
    assess = assess[assess['ASES_NM']==ases_nm]
    assess_cnt = np.zeros(programs.shape[0])
    assess_cnt = list(assess_cnt)
    for index, row in programs.iterrows():
        assess_pat = assess[(assess['EMPI']==row['EMPI']) &
                             (assess['ASES_DT'] > (row['prog_create_date']-timedelta(days=10))) &
                             (assess['ASES_DT'] < (row['prog_create_date']+timedelta(days=35)))]
        assess_cnt[index] = assess_pat.shape[0]
    return assess_cnt


# In[ ]:


def find_assessments_during_program_chf(programs, assess):
    assess_cnt = np.zeros(programs.shape[0])
    assess_cnt = list(assess_cnt)
    for index, row in programs.iterrows():
        assess_pat = assess[(assess['EMPI']==row['EMPI']) &
                             (assess['ASES_DT'] > (row['prog_create_date']-timedelta(days=1))) &
                             (assess['ASES_DT'] < (row['prog_create_date']+timedelta(days=90)))]
        assess_cnt[index] = assess_pat.shape[0]
    return assess_cnt


# In[ ]:


def find_assessments_during_program_all(programs, assess, list_of_assessments):
    assess_cnt = np.zeros(len(list_of_assessments))
    assess_cnt = list(assess_cnt)
    for index, ases in enumerate(list_of_assessments):
        print(ases)
        count = 0
        for index2, row2 in programs.iterrows():
            assess_pat = assess[(assess['EMPI']==row2['EMPI']) &
                                 (assess['ASES_DT'] > row2['prog_create_date']) &
                                 (assess['ASES_DT'] < row2['prog_end_date']) &
                                 (assess['ASES_NM']==ases)]
            count += assess_pat.shape[0]
        assess_cnt[index] = count
    return assess_cnt


# In[ ]:


def count_ases(assess_all):
    counts_ases = []
    for index, value in enumerate(list_o_ases):
        print(value)
        count = 0
        for index2, row in assess_all.iterrows():
            if row['ASES_NM']==value:
                count+=1
        counts_ases.append(count)
    return counts_ases


# In[ ]:


def adms_to_one_zero_v2(ases):
    ases_yn = []
    for num in ases:
        if num>0:
            ases_yn.append(1)
        else:
            ases_yn.append(0)
    return ases_yn


# In[ ]:


def feature_from_assessment_text(programs, assess):
    assess = assess[(assess['ASES_NM']=='TOC Post Discharge Outreach') & 
                    (assess['ANSR_TXT']=='Success') |
                    (assess['ANSR_TXT']=='Continue TOC')]
    values_return = np.zeros(programs.shape[0])
    values_return = list(values_return)
    for index, row in programs.iterrows():
        assess_pat = assess[(assess['EMPI']==row['EMPI']) &
                             (assess['ASES_DT'] > (row['prog_create_date']-timedelta(days=10))) &
                             (assess['ASES_DT'] < (row['prog_create_date']+timedelta(days=35)))]
        individual = list(set(assess_pat['PTNT_ASES_DK']))
        print(individual)
        count=0
        for assess in individual:
            touches = assess_pat[assess_pat['PTNT_ASES_DK']==assess]
            for index2, row2 in touches.iterrows():
                if (row2['ANSR_TXT']=='Success') | (row2['ANSR_TXT']=='Continue TOC'):
                    count+=1
        values_return[index]=count
    return values_return


# In[ ]:


## function that counts the number of admits in a window AFTER a program begins
def get_adm_after(programs, admissions, window_size=90):
    admits_in_window = list(np.zeros(programs.shape[0]))
    beddays_in_window = list(np.zeros(programs.shape[0]))
    for index, row in programs.iterrows():
        admit_pat = admissions[admissions['EMPI']==row['EMPI']]
        count = 0
        beddays = 0
        for index2, row2 in admit_pat.iterrows():
            if (row2['admit_date'] < (row['prog_create_date']+timedelta(days=window_size))) & (row2['admit_date'] > row['prog_create_date']):
                count+=1
                beddays += row2['WTD_DAY_CT']
        admits_in_window[index] = count
        beddays_in_window[index] = beddays
    return admits_in_window, beddays_in_window


# In[ ]:


## function that counts the number of admits in a window BEFORE a program begins
def get_adm_before(programs, admissions, window_size=90):
    admits_in_window = list(np.zeros(programs.shape[0]))
    beddays_in_window = list(np.zeros(programs.shape[0]))
    for index, row in programs.iterrows():
        admit_pat = admissions[admissions['EMPI']==row['EMPI']]
        count = 0
        beddays = 0
        for index2, row2 in admit_pat.iterrows():
            if (row2['admit_date'] > (row['prog_create_date']-timedelta(days=window_size))) & (row2['admit_date'] > (row['prog_create_date']-timedelta(days=10))):
                count+=1
                beddays += row2['WTD_DAY_CT']
        admits_in_window[index] = count
        beddays_in_window[index] = beddays
    return admits_in_window, beddays_in_window


# In[ ]:


## function that counts the number of admits in a window AFTER a program begins
def get_adm_after_TOC(programs, admissions, window_size=30):
    admits_in_window = list(np.zeros(programs.shape[0]))
    for index, row in programs.iterrows():
        admit_pat = admissions[admissions['EMPI']==row['EMPI']]
        count = 0
        for index2, row2 in admit_pat.iterrows():
            if (row2['admit_date'] < (row['index_date']+timedelta(days=window_size))) & (row2['admit_date'] > row['index_date']):
                count+=1
        admits_in_window[index] = count
    return admits_in_window


# In[ ]:


def adms_to_one_zero(progs, field):
    adm_yn = []
    for index, row in progs.iterrows():
        if row[field]>0:
            adm_yn.append(1)
        else:
            adm_yn.append(0)
    return adm_yn


# In[ ]:


def plotroc(TPR, FPR):
    roc_auc = auc(TPR, FPR)
    plt.figure()
    lw = 2
    plt.plot(TPR, FPR, color='darkorange',
             lw=lw, label="ROC curve area = {0:0.4f}".format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


# In[ ]:


def bootstrap_ci_coefficients(X_train, y_train, num_bootstraps, class_weight=class_weight):
    bootstrap_estimates = []
    for i in np.arange(num_bootstraps):
        sample_index = np.random.choice(range(0, len(y_train)), len(y_train))
        X_samples = X_train[sample_index]
        y_samples = y_train[sample_index]
        lm = linear_model.LogisticRegression(class_weight=class_weight, penalty='l1', C=1)
        lm.fit(X_samples, y_samples)
        bootstrap_estimates.append(lm.coef_[0])
    bootstrap_estimates = np.asarray(bootstrap_estimates)
    return bootstrap_estimates


# In[ ]:


def bootstrap_ci_coefficients_lin(X_train, y_train, num_bootstraps):
    bootstrap_estimates = []
    for i in np.arange(num_bootstraps):
        sample_index = np.random.choice(range(0, len(y_train)), len(y_train))
        X_samples = X_train[sample_index]
        y_samples = y_train[sample_index]
        lm = linear_model.LinearRegression()
        lm.fit(X_samples, y_samples)
        bootstrap_estimates.append(lm.coef_)
    bootstrap_estimates = np.asarray(bootstrap_estimates)
    return bootstrap_estimates


# ## Read in Data

# In[ ]:


## READ In Programs, Admissions, and Lace data
## Could have script run this
progs = pd.read_csv('data/Program_Patdim.csv', sep='|', low_memory=False)


# In[ ]:


admits = pd.read_csv('data/admissions.csv', sep='|', low_memory=False)


# In[ ]:


assess = pd.read_csv('data/assessments.csv.sql', sep='|', low_memory=False)


# In[ ]:


## Cleaning, field removal, and data type changes


# ## Clean Data

# In[ ]:


## convert dates to date time
progs['prog_create_date'] = pd.to_datetime(progs['ASGN_TMS'])
progs['date_of_birth'] = pd.to_datetime(progs['DOB'])
progs['prog_end_date'] = pd.to_datetime(progs['END_TMS'])


# In[ ]:


## replace sex with is_male
opt_in = {'Admit to Facility':1, 'Discharge to Custodial':1, 'Discharged to Hospice':1, 
          'Discharged to PCP':1, 'Goals Met':1, 'Referred to other program':1, 
          'Refused':0, 'Unable to Reach':0, 'Barriers to participation':0, 'Expired':np.nan, 
         'Criteria not met':np.nan, 'Disenrolled from HP/Medical Group':np.nan, '6 months/1 year post transplant':np.nan,
       'Pharmacy - physician recommended':np.nan}
progs['is_optin'] = progs['PRGM_STOP_RSN'].replace(opt_in)
# progs['is_male'].fillna(0, inplace=True)


# In[ ]:


## replace date_of _ birth with age
progs['age'] = datetime.now() - progs['date_of_birth']
progs['age'] = progs['age'] / timedelta(days=1) / 365
progs['age'].fillna(float(progs['age'].mean()), inplace=True)


# In[ ]:


new_sex = {'F':0, 'M':1}
progs['is_male'] = progs['Sex'].replace(new_sex)
progs['is_male'].fillna(0, inplace=True)


# In[ ]:


## drop unneeded columns
prog_cols_drop = ['PTNT_DK', 'DOB', 'Sex', 'ASGN_TMS', 'END_TMS', 'TNT_MKT_BK', 'date_of_birth', 'PRGM_STS']
progs = progs.drop(prog_cols_drop, axis=1)


# In[ ]:


progs = progs[(progs['prog_create_date']>'2018-04-01') & (progs['prog_create_date']<'2018-09-01')]


# In[ ]:


progs['EMPI'] = progs['EMPI'].astype(int)


# In[ ]:


progs = progs.reset_index()


# In[ ]:


## convert object columns to categoricals
for col in ['RGON_NM', 'HP_NM', 'LOB_SUB_CGY', 'PRGM_NM', 'PRGM_STOP_RSN']:
    progs[col] = progs[col].astype('category')


# In[ ]:


admits['EMPI'].fillna(999999999, inplace=True)

admits['EMPI'] = admits['EMPI'].astype(int)


# In[ ]:


## convert dates to date time
admits['admit_date'] = pd.to_datetime(admits['AdmitDt'])
admits['discharge_date'] = pd.to_datetime(admits['DischDt'])


# In[ ]:


## replace Model with is_group
new_Model = {'GROUP':1, 'IPA':0, 'JOINT VENTURE':0, 'OPEN ACCESS':0, 'OTHER':0}
admits['is_group'] = admits['Model'].replace(new_Model)
admits['is_group'].fillna(0, inplace=True)


# In[ ]:


## drop unneeded columns
admits_cols_drop = ['Model', 'PCP', 'MM', 'MRN', 'LastName',
       'FirstName', 'AdmitDt', 'DischDt']
admits = admits.drop(admits_cols_drop, axis=1)


# In[ ]:


## REmove assessments with weird filler date '2917-12-26 00:00:00.000'
assess = assess[assess['ASES_DT']!='2917-12-26 00:00:00.000']


# In[ ]:


assess['ASES_DT'] = pd.to_datetime(assess['ASES_DT'])
assess['EFF_FM_TS'] = pd.to_datetime(assess['EFF_FM_TS'])


# In[ ]:


new_text = {'Successful (enter number you\'re calling)':'Success'}
assess['ANSR_TXT'] = assess['ANSR_TXT'].replace(new_text)


# In[ ]:


## change objects to categories
cols_to_category = ['REGION', 'SITE', 'LOB', 'Acuity', 'Facility',
       'RefType', 'DayType', 'AdmissionType', 'DischDx1', 'DischDx1Desc', 'SurgeryPx', 'DISPOSITION', 'ACSA_CAT']
for col in cols_to_category:
    admits[col] = admits[col].astype('category')


# In[ ]:


## find the set of all PTNT_ASES_DK, will be used to loop through


# In[ ]:


## create new columns using functions and the admits data


# In[ ]:


admits_all = admits


# In[ ]:


## only keep acute admissions
admits = admits[admits['Acuity']=='ACUTE']


# In[ ]:


readmits = admits[admits['READMIT_N']==1]


# ## Feature Engineering

# In[ ]:


index_dates = find_index_admit(progs, admits_all)
progs['index_date'] = index_dates
progs['index_date'] = pd.to_datetime(progs['index_date'])


# In[ ]:


first_admissions = find_first_admission_after_enroll(progs, admits)
progs['frst_adm_aftr_enrl'] = first_admissions
progs['frst_adm_aftr_enrl'] = pd.to_datetime(progs['frst_adm_aftr_enrl'])


# In[ ]:


LACE = assess[assess['ASES_NM']=='LACE']


# In[ ]:


lace_scores = find_lace_prior_to_enroll(progs, LACE)
progs['lace_score'] = lace_scores


# In[ ]:


## create count and y/n columns
cm_toc_ases = find_assessments_during_program(progs, assess, ases_nm='MCG - DMG - Post-Hospitalization Follow-Up')
cm_toc_ases2 = adms_to_one_zero_v2(cm_toc_ases)
progs['cnt_toc_cm_touch'] = cm_toc_ases
progs['toc_cm_touch_yn'] = cm_toc_ases2


# In[ ]:


# ## subset assessment to only include questions where the answer = "Success"
# success_ases = assess[assess['ANSR_TXT']=='Success']
# TOCPDO_cnt = find_assessments_during_program(progs, success_ases)
# progs['success_TOCPDO_cnt'] = TOCPDO_cnt


# We hope to include "Medication Reconciliation" and "Post Hospitalization Follow-up" to the model, but currently these assessments only date back to mid-August, and therefore don't cover patients in TOC since April.

# In[ ]:


# docrev_dissum = feature_from_assessment_text(progs, assess, ases_nm='TOC Post Discharge Outreach',
#                                  qstn_txt='Indicate the Documents Reviewed',
#                                 ansr_txt='Discharge Summary')
# progs['docrev_dissum'] = docrev_dissum


# In[ ]:


# docrev_histphys = feature_from_assessment_text(progs, assess, ases_nm='TOC Post Discharge Outreach',
#                                  qstn_txt='Indicate the Documents Reviewed',
#                                 ansr_txt='History and Physical')
# progs['docrev_histphys'] = docrev_histphys


# In[ ]:


# docrev_medrec_ehr = feature_from_assessment_text(progs, assess, ases_nm='TOC Post Discharge Outreach',
#                                  qstn_txt='Indicate the Documents Reviewed',
#                                 ansr_txt='Medication reconciliation completed in TWEHR')
# progs['docrev_medrec_ehr'] = docrev_medrec_ehr


# In[ ]:


# docrev_meds = feature_from_assessment_text(progs, assess, ases_nm='TOC Post Discharge Outreach',
#                                  qstn_txt='Indicate the Documents Reviewed',
#                                 ansr_txt='Medications')
# progs['docrev_meds'] = docrev_meds


# In[ ]:


# docrev_PCPrecs = feature_from_assessment_text(progs, assess, ases_nm='TOC Post Discharge Outreach',
#                                  qstn_txt='Indicate the Documents Reviewed',
#                                 ansr_txt='PCP Records')
# progs['docrev_PCPrecs'] = docrev_PCPrecs


# In[ ]:


# docrev_problist = feature_from_assessment_text(progs, assess, ases_nm='TOC Post Discharge Outreach',
#                                  qstn_txt='Indicate the Documents Reviewed',
#                                 ansr_txt='Problems List')
# progs['docrev_problist'] = docrev_problist


# In[ ]:


# docrev_refhist = feature_from_assessment_text(progs, assess, ases_nm='TOC Post Discharge Outreach',
#                                  qstn_txt='Indicate the Documents Reviewed',
#                                 ansr_txt='Referral History')
# progs['docrev_refhist'] = docrev_refhist


# In[ ]:


# no_part_CM = feature_from_assessment_text(progs, assess, ases_nm='TOC Post Discharge Outreach',
#                                  qstn_txt='Patient is willing to participate in Care Management?',
#                                 ansr_txt='No')
# progs['no_part_CM'] = no_part_CM


# In[ ]:


# no_part_DM = feature_from_assessment_text(progs, assess, ases_nm='TOC Post Discharge Outreach',
#                                  qstn_txt='Patient is willing to participate in Care or Disease Management and agrees to the recommended contact interval. The program was explained to the patient and they understand it is optional and that they can opt out at any time.',
#                                 ansr_txt='No')
# progs['no_part_DM'] = no_part_DM


# In[ ]:


# agree_care_plan = feature_from_assessment_text(progs, assess, ases_nm='TOC Post Discharge Outreach',
#                                  qstn_txt='Patient verbalized agreement to Care Plan',
#                                 ansr_txt='Yes')
# progs['agree_care_plan'] = agree_care_plan


# In[ ]:


##medrec_cnt = find_assessments_during_program(progs,assess, ases_nm='CM Medication Reconciliation')
##progs['medrec_cnt']=medrec_cnt


# In[ ]:


##phfu_cnt = find_assessments_during_program(progs ,assess, ases_nm='Post-Hospitalization Follow-up')
##progs['phfu_cnt']=phfu_cnt


# In[ ]:


## calc number of admits that occur within 30 day window after program begins
thirty_day_after_adm, thirty_day_after_bd  = get_adm_after(progs, admits, window_size=30)
progs['adm_30_after'] = thirty_day_after_adm
progs['bd_30_after'] = thirty_day_after_bd


# In[ ]:


## calc number of admits that occur within 90 day window after program begins
ninety_day_after_adm, ninety_day_after_bd = get_adm_after(progs, admits, window_size=90)
progs['adm_90_after'] = ninety_day_after_adm
progs['bd_90_after'] = ninety_day_after_bd


# In[ ]:


## calc number of readmits that occur within 90 day window after program begins
ninety_day_after_readm, ninety_day_after_bd = get_adm_after(progs, readmits, window_size=90)
progs['readm_90_after'] = ninety_day_after_readm


# In[ ]:


thirty_day_after_TOC = get_adm_after_TOC(progs, admits)
progs['adm_30_after_TOC'] = thirty_day_after_TOC


# In[ ]:


thirty_day_after_TOC_re = get_adm_after_TOC(progs, readmits)
progs['readm_30_after_TOC'] = thirty_day_after_TOC_re


# In[ ]:


## function that counts relevant touches during a care management program
#def count_touches_that_count(programs, assessments):
#    touches_that_count = list(np.zeros(programs.shape[0]))
#    for index, row in programs.iterrows():
#        assess


# In[ ]:


# ## calc number of admits that occur within 30 day window before program begins
# thirty_day_before_adm, thirty_day_before_bd  = get_adm_before(progs, admits, window_size=30)
# progs['adm_30_before'] = thirty_day_before_adm
# progs['bd_30_before'] = thirty_day_before_bd


# In[ ]:


## calc number of admits that occur within 90 day window before program begins
ninety_day_before_adm , ninety_day_before_bd = get_adm_before(progs, admits)
progs['adm_90_before'] = ninety_day_before_adm
progs['bd_90_before'] = ninety_day_before_bd


# In[ ]:


## calc number of admits that occur within 90 day window before program begins
ninety_day_before_readm , ninety_day_before_bd = get_adm_before(progs, readmits)
progs['adm_90_before'] = ninety_day_before_readm


# In[ ]:


progs['time_to_enroll'] = progs['prog_create_date']-progs['index_date']
progs['time_to_enroll'] = progs['time_to_enroll']/ timedelta(days=1)


# In[ ]:


# progs['time_to_enroll'].fillna(float(progs['time_to_enroll'].mean()), inplace=True)


# In[ ]:


progs['prog_duration'] = progs['prog_end_date']-progs['prog_create_date']
progs['prog_duration'] = progs['prog_duration']/ timedelta(days=1)


# In[ ]:


# progs['prog_duration'].fillna(float(progs['prog_duration'].median()), inplace=True)


# In[ ]:


progs['index_to_next_days'] = progs['frst_adm_aftr_enrl']-progs['index_date']
progs['index_to_next_days'] = progs['index_to_next_days']/ timedelta(days=1)


# In[ ]:


adm_yn = adms_to_one_zero(progs, 'adm_30_after_TOC')


# In[ ]:


progs['is_30_TOC_adm'] = adm_yn


# In[ ]:


readm_yn = adms_to_one_zero(progs, 'readm_30_after_TOC')


# In[ ]:


progs['is_30_TOC_readm'] = readm_yn


# ## Exploratory Data Analysis

# In[ ]:


# fighist = plt.figure(figsize=(12,6))
# ax1 = fighist.add_subplot(111)
# ax1.set_title('Histogram, Days Program Duration')
# #ax1.hist(np.array(progs[progs['PRGM_NM']=='Transitions of Care - Post Discharge']['time_to_enroll'].dropna()), bins = 100, alpha = 0.4, density=1, label='TOC')
# ax1.hist(np.array(progs[progs['PRGM_NM']=='DM - CLD']['prog_duration'].dropna()), bins = 150, alpha = 0.6, density=1, label='CLD')
# ax1.hist(np.array(progs[progs['PRGM_NM']=='DM - HF']['prog_duration'].dropna()), bins = 150, alpha = 0.6, density=1, label='HF')
# ax1.set_ylabel('Percent of Patients')
# ax1.set_xlabel('Days in Program')
# ax1.set_xlim(left=-5, right=150)
# ax1.legend();


# In[ ]:


# fighist = plt.figure(figsize=(12,6))
# ax1 = fighist.add_subplot(111)
# ax1.set_title('Histogram, time to enroll post discharge')
# #ax1.hist(np.array(progs[progs['PRGM_NM']=='Transitions of Care - Post Discharge']['time_to_enroll'].dropna()), bins = 100, alpha = 0.4, density=1, label='TOC')
# ax1.hist(np.array(progs[progs['PRGM_NM']=='DM - CLD']['time_to_enroll'].dropna()), bins = 40, alpha = 0.6, density=1, label='CLD')
# ax1.hist(np.array(progs[progs['PRGM_NM']=='DM - HF']['time_to_enroll'].dropna()), bins = 40, alpha = 0.6, density=1, label='HF')
# ax1.set_ylabel('Percent of Patients')
# ax1.set_xlabel('days to enrollment')
# ax1.legend();


# In[ ]:


# fighist_TOC = plt.figure(figsize=(12,6))
# ax1 = fighist_TOC.add_subplot(111)
# ax1.set_title('Histogram, Program Duration')
# ax1.hist(np.array((progs[(progs['PRGM_NM']=='Transitions of Care - Post Discharge') &
#                   (progs['is_optin']==0)]['prog_duration'].dropna())), bins = 100, alpha = 0.6, density=1, color='grey', label='Opt-out')
# ax1.hist(np.array((progs[(progs['PRGM_NM']=='Transitions of Care - Post Discharge') &
#                   (progs['is_optin']==1)]['prog_duration'].dropna())), bins = 100, alpha = 0.6, density=1, color='green', label='Opt-in')
# ax1.set_ylabel('Percent of Patients')
# ax1.set_xlabel('days in program')
# ax1.set_xlim(left=-5, right=50)
# ax1.legend();


# In[ ]:


# fighist_TOC = plt.figure(figsize=(12,6))
# ax1 = fighist_TOC.add_subplot(111)
# ax1.set_title('Histogram, LACE Scores by Program Participation, TOC')
# ax1.hist(np.array((progs[(progs['PRGM_NM']=='Transitions of Care - Post Discharge') &
#                   (progs['is_optin']==0)]['lace_score'].dropna())), 
#          bins = 18, alpha = 0.6, density=1, color='grey', label='Opt-out')
# ax1.hist(np.array((progs[(progs['PRGM_NM']=='Transitions of Care - Post Discharge') &
#                   (progs['is_optin']==1)]['lace_score'].dropna())), 
#          bins = 18, alpha = 0.6, density=1, color='green', label='Opt-in')
# ax1.set_ylabel('Percent of Patients')
# ax1.set_xlabel('Lace Score')
# ax1.set_xlim(left=0, right=20)
# ax1.legend();


# In[ ]:


# fighist_TOC = plt.figure(figsize=(12,6))
# ax1 = fighist_TOC.add_subplot(111)
# ax1.set_title('Histogram, Days to Enrollment, TOC')
# ax1.hist(np.array((progs[(progs['PRGM_NM']=='Transitions of Care - Post Discharge') &
#                   (progs['is_optin']==0)]['time_to_enroll'].dropna())), bins = 150, alpha = 0.6, density=1, color='grey', label='Opt-out')
# ax1.hist(np.array((progs[(progs['PRGM_NM']=='Transitions of Care - Post Discharge') &
#                   (progs['is_optin']==1)]['time_to_enroll'].dropna())), bins = 150, alpha = 0.6, density=1, color='green', label='Opt-in')
# ax1.set_ylabel('Percent of Patients')
# ax1.set_xlabel('Days to Enrollment')
# ax1.set_xlim(left=0, right=50)
# ax1.legend();


# In[ ]:


# fighist_TOC = plt.figure(figsize=(12,6))
# ax1 = fighist_TOC.add_subplot(111)
# ax1.set_title('Histogram, time to enroll (TOC)')
# ax1.hist(np.array(progs[progs['PRGM_NM']=='Transitions of Care - Post Discharge']['time_to_enroll'].dropna()), bins = 200, alpha = .6, density=1, color='green', label='TOC')
# ax1.set_ylabel('Percent of Patients')
# ax1.set_xlabel('days to enrollment')
# ax1.set_xlim(left=0, right=40)
# ax1.legend();


# In[ ]:


# progs[progs['PRGM_NM']=='Transitions of Care - Post Discharge'].pivot_table(values='adm_30_after_TOC', index='PRGM_STOP_RSN', aggfunc=['count','mean'], dropna=True)


# In[ ]:


## Understand relationship between LACE and time to enroll


# In[ ]:


# progs[progs['PRGM_NM']=='Transitions of Care - Post Discharge'].pivot_table(values='adm_30_after_TOC', index='lace_score', columns='is_optin', aggfunc=['count','mean'], dropna=True)


# In[ ]:


# ## graph violinplots of optin vs. optout LACE Scores.
# fighist_lace_age = plt.figure(figsize=(12,6))
# ax1 = fighist_lace_age.add_subplot(111)
# sns.boxplot(ax=ax1, x="lace_score", y='age', data=progs[progs['PRGM_NM']=='Transitions of Care - Post Discharge'])


# In[ ]:


# ## graph violinplots of optin vs. optout LACE Scores.
# sns.violinplot(x="is_optin", y='lace_score', data=progs[progs['PRGM_NM']=='Transitions of Care - Post Discharge'])


# In[ ]:


## use logisic regression to test whether opt_in has impact on readmit_30_days_TOC, controlling for other variables.


# ## TOC

# In[ ]:


progs_toc = progs


# In[ ]:


# ## Two variables, plotting y's
# fig1 = plt.figure(figsize=(12,10))
# ax1 = fig1.add_subplot(111)
# ax1.scatter(progs_toc['lace_score'], progs_toc['bd_90_before'], color='b')
# #ax1.scatter(X_horses[:,0], X_horses[:,1], color='r', label='horses')
# ax1.legend(shadow=True, fontsize='xx-large')
# #ax1.set_xlabel('Weight (lb)',fontsize=font_size)
# #ax1.set_ylabel('Height (in)',fontsize=font_size)
# #ax1.set_title('Horse or dog?',fontsize=font_size)
# plt.show()


# In[ ]:


progs_toc = progs_toc[progs_toc['PRGM_NM']=='Transitions of Care - Post Discharge']


# In[ ]:


progs_toc_w_lace = progs_toc[(progs_toc['lace_score'].isna()==False) & (progs_toc['prog_create_date']<'2018-08-15')].reset_index()


# In[ ]:


class_weight={0: 0.5, 1: 2}


# In[ ]:


progs_toc_ent = pd.get_dummies(pd.Series(list(progs_toc_w_lace['ENT_TYPE'])))


# In[ ]:


progs_toc_lob = pd.get_dummies(pd.Series(list(progs_toc_w_lace['LOB_SUB_CGY'])))


# In[ ]:


progs_toc_rgn = pd.get_dummies(pd.Series(list(progs_toc_w_lace['RGON_NM'])))


# In[ ]:


progs_toc_w_lace = pd.concat([progs_toc_w_lace, progs_toc_lob, progs_toc_ent, progs_toc_rgn], axis=1)


# In[ ]:


X = np.array(progs_toc_w_lace[['toc_cm_touch_yn','age', 'is_male', 'bd_90_before', 'lace_score', 'time_to_enroll',
                               'prog_duration', 'COMMERCIAL', 'MEDI-CAL', 'IPA', 'LA/DOWNTOWN', 'LONG BEACH', 'MAGAN',
                               'ORANGE COUNTY', 'SAN FERNANDO VALLEY', 'SOUTH BAY', 'VILLAGE HEALTH']])


# In[ ]:


imput = Imputer(strategy='median')


# In[ ]:


X = imput.fit_transform(X)


# #### Readmissions

# In[ ]:


y_read_yn = np.array(progs_toc_w_lace['is_30_TOC_readm'])


# In[ ]:


X_read_train, X_read_test, y_read_yn_train, y_read_yn_test = train_test_split(X, y_read_yn, test_size=.25)


# In[ ]:


model_read = linear_model.LogisticRegression(C=1, penalty='l1', class_weight=class_weight)


# In[ ]:


model_read.fit(X_read_train, y_read_yn_train)


# In[ ]:


model_read.coef_


# In[ ]:


y_read_yn_preds = model_read.predict_proba(X_read_test)[:,1]


# In[ ]:


y_read_yn_preds_act = model_read.predict(X_read_test)


# In[ ]:


y_read_yn_test.mean()


# In[ ]:


y_read_yn_preds_act.mean()


# In[ ]:


read_yn_bootstraps = bootstrap_ci_coefficients(X_read_train, y_read_yn_train, 2000)


# In[ ]:


read_yn_bootstraps = pd.DataFrame(read_yn_bootstraps, columns=['toc_cm_touch_yn','age', 'is_male', 'bd_90_before', 'lace_score', 'time_to_enroll',
                               'prog_duration', 'COMMERCIAL', 'MEDI-CAL', 'IPA', 'LA/DOWNTOWN', 'LONG BEACH', 'MAGAN',
                               'ORANGE COUNTY', 'SAN FERNANDO VALLEY', 'SOUTH BAY', 'VILLAGE HEALTH'])


# In[ ]:


fig, axes = plt.subplots(4,4, figsize=(16,16))
col_names = ['toc_cm_touch_yn','age', 'is_male', 'bd_90_before', 'lace_score', 'time_to_enroll',
                               'prog_duration', 'COMMERCIAL', 'MEDI-CAL', 'IPA', 'LA/DOWNTOWN', 'LONG BEACH', 'MAGAN',
                               'ORANGE COUNTY', 'SAN FERNANDO VALLEY', 'SOUTH BAY', 'VILLAGE HEALTH']

for m, ax in zip(col_names, axes.flatten()):
    ax.hist(read_yn_bootstraps[m], bins=50)
    ax.set_title(m)


# In[ ]:


coefs = model_read.coef_


# In[ ]:


pd.concat([pd.Series(['toc_cm_touch_yn','age', 'is_male', 'bd_90_before', 'lace_score', 'time_to_enroll',
                               'prog_duration', 'COMMERCIAL', 'MEDI-CAL', 'IPA', 'LA/DOWNTOWN', 'LONG BEACH', 'MAGAN',
                               'ORANGE COUNTY', 'SAN FERNANDO VALLEY', 'SOUTH BAY', 'VILLAGE HEALTH']), pd.Series(np.exp(coefs)[0])], axis=1, )


# In[ ]:


inter = model_read.intercept_


# In[ ]:


inter


# ['toc_cm_touch_yn','age', 'is_male', 'bd_90_before', 'lace_score', 'time_to_enroll',
#                                'prog_duration', 'COMMERCIAL', 'MEDI-CAL', 'IPA', 'LA/DOWNTOWN', 'LONG BEACH', 'MAGAN',
#                                'ORANGE COUNTY', 'SAN FERNANDO VALLEY', 'SOUTH BAY', 'VILLAGE HEALTH']

# In[ ]:


np.exp((coefs[0] * [1,70,1,1,10,2,30,0,1,1,1,0,0,0,0,0,0]).sum() + inter)


# In[ ]:


TPR, FPR, thresholds = roc_curve(y_read_yn_test, y_read_yn_preds, pos_label=None, sample_weight=None, drop_intermediate=True)


# In[ ]:


plotroc(TPR, FPR)


# #### Admissions

# In[ ]:


y_ad_yn = np.array(progs_toc_w_lace['is_30_TOC_adm'])


# In[ ]:


X_ad_train, X_ad_test, y_ad_yn_train, y_ad_yn_test = train_test_split(X, y_ad_yn, test_size=.15)


# In[ ]:


model_ad = linear_model.LogisticRegression(C=1, penalty='l1', class_weight=class_weight)


# In[ ]:


model_ad.fit(X_ad_train, y_ad_yn_train)


# In[ ]:


model_ad.coef_


# In[ ]:


y_ad_yn_preds = model_ad.predict_proba(X_ad_test)[:,1]


# In[ ]:


y_ad_yn_preds_act = model_ad.predict(X_ad_test)


# In[ ]:


y_ad_yn_test.mean()


# In[ ]:


y_ad_yn_preds_act.mean()


# In[ ]:


ad_yn_bootstraps = bootstrap_ci_coefficients(X_ad_train, y_ad_yn_train, 2000)


# In[ ]:


ad_yn_bootstraps = pd.DataFrame(ad_yn_bootstraps, columns=['toc_cm_touch_yn','age', 'is_male', 'bd_90_before', 'lace_score', 'time_to_enroll',
                               'prog_duration', 'COMMERCIAL', 'MEDI-CAL', 'IPA', 'LA/DOWNTOWN', 'LONG BEACH', 'MAGAN',
                               'ORANGE COUNTY', 'SAN FERNANDO VALLEY', 'SOUTH BAY', 'VILLAGE HEALTH'])


# In[ ]:


fig, axes = plt.subplots(4,4, figsize=(16,16))
col_names = ['toc_cm_touch_yn','age', 'is_male', 'bd_90_before', 'lace_score', 'time_to_enroll',
                               'prog_duration', 'COMMERCIAL', 'MEDI-CAL', 'IPA', 'LA/DOWNTOWN', 'LONG BEACH', 'MAGAN',
                               'ORANGE COUNTY', 'SAN FERNANDO VALLEY', 'SOUTH BAY', 'VILLAGE HEALTH']

for m, ax in zip(col_names, axes.flatten()):
    ax.hist(ad_yn_bootstraps[m], bins=50)
    ax.set_title(m)


# is_male positive
# bd positive
# lace positive
# time to enroll neg
# medical positive
# la almost positive
# oc neg
# sfv positive
# sb positive

# In[ ]:


ad_coefs = model_ad.coef_


# In[ ]:


pd.concat([pd.Series(['toc_cm_touch_yn','age', 'is_male', 'bd_90_before', 'lace_score', 'time_to_enroll',
                               'prog_duration', 'COMMERCIAL', 'MEDI-CAL', 'IPA', 'LA/DOWNTOWN', 'LONG BEACH', 'MAGAN',
                               'ORANGE COUNTY', 'SAN FERNANDO VALLEY', 'SOUTH BAY', 'VILLAGE HEALTH']), pd.Series(np.exp(ad_coefs)[0])], axis=1, )


# In[ ]:


inter = model_read.intercept_


# In[ ]:


inter


# ['toc_cm_touch_yn','age', 'is_male', 'bd_90_before', 'lace_score', 'time_to_enroll',
#                                'prog_duration', 'COMMERCIAL', 'MEDI-CAL', 'IPA', 'LA/DOWNTOWN', 'LONG BEACH', 'MAGAN',
#                                'ORANGE COUNTY', 'SAN FERNANDO VALLEY', 'SOUTH BAY', 'VILLAGE HEALTH']

# In[ ]:


np.exp((ad_coefs[0] * [1,70,0,0,5,2,30,0,0,1,1,0,0,0,0,0,0]).sum() + inter)


# In[ ]:


ad_TPR, ad_FPR, ad_thresholds = roc_curve(y_ad_yn_test, y_ad_yn_preds, pos_label=None, sample_weight=None, drop_intermediate=True)


# In[ ]:


plotroc(ad_TPR, ad_FPR)


# #### 30 Day Bed Days

# In[ ]:


y_30_bd = np.array(progs_toc_w_lace['bd_30_after'])


# In[ ]:


X_bd_train, X_bd_test, y_bd_train, y_bd_test = train_test_split(X, y_30_bd, test_size=.25)


# In[ ]:


model_bd = linear_model.LinearRegression()


# In[ ]:


model_bd.fit(X_bd_train, y_bd_train)


# In[ ]:


bd_coefs = model_bd.coef_


# In[ ]:


y_bd_preds = model_bd.predict(X_bd_test)


# In[ ]:


y_bd_test.mean()


# In[ ]:


y_bd_preds.mean()


# In[ ]:


bd_bootstraps = bootstrap_ci_coefficients_lin(X_bd_train, y_bd_train, 2000)


# In[ ]:


bd_bootstraps = pd.DataFrame(bd_bootstraps, columns=['toc_cm_touch_yn','age', 'is_male', 'bd_90_before', 'lace_score', 'time_to_enroll',
                               'prog_duration', 'COMMERCIAL', 'MEDI-CAL', 'IPA', 'LA/DOWNTOWN', 'LONG BEACH', 'MAGAN',
                               'ORANGE COUNTY', 'SAN FERNANDO VALLEY', 'SOUTH BAY', 'VILLAGE HEALTH'])


# In[ ]:


fig, axes = plt.subplots(4,4, figsize=(16,16))
col_names = ['toc_cm_touch_yn','age', 'is_male', 'bd_90_before', 'lace_score', 'time_to_enroll',
                               'prog_duration', 'COMMERCIAL', 'MEDI-CAL', 'IPA', 'LA/DOWNTOWN', 'LONG BEACH', 'MAGAN',
                               'ORANGE COUNTY', 'SAN FERNANDO VALLEY', 'SOUTH BAY', 'VILLAGE HEALTH']

for m, ax in zip(col_names, axes.flatten()):
    ax.hist(bd_bootstraps[m], bins=50)
    ax.set_title(m)


# In[ ]:


pd.concat([pd.Series(['toc_cm_touch_yn','age', 'is_male', 'bd_90_before', 'lace_score', 'time_to_enroll',
                               'prog_duration', 'COMMERCIAL', 'MEDI-CAL', 'IPA', 'LA/DOWNTOWN', 'LONG BEACH', 'MAGAN',
                               'ORANGE COUNTY', 'SAN FERNANDO VALLEY', 'SOUTH BAY', 'VILLAGE HEALTH']), pd.Series(bd_coefs)], axis=1, )


# In[ ]:


inter = model_read.intercept_


# In[ ]:


inter


# ['toc_cm_touch_yn','age', 'is_male', 'bd_90_before', 'lace_score', 'time_to_enroll',
#                                'prog_duration', 'COMMERCIAL', 'MEDI-CAL', 'IPA', 'LA/DOWNTOWN', 'LONG BEACH', 'MAGAN',
#                                'ORANGE COUNTY', 'SAN FERNANDO VALLEY', 'SOUTH BAY', 'VILLAGE HEALTH']

# In[ ]:


progs_toc_w_lace.pivot_table(['preds', 'is_30_TOC_adm'], index='LOB_SUB_CGY', aggfunc=['mean'])


# In[ ]:


progs_toc_w_lace.pivot_table(['preds', 'is_30_TOC_adm'], index='RGON_NM', aggfunc='mean')


# In[ ]:


progs_toc_w_lace['preds'].mean()


# In[ ]:


progs_toc_w_lace['is_30_TOC_adm'].mean()


# In[ ]:


gb_mod = GradientBoostingClassifier(learning_rate=0.05, max_depth=8, n_estimators=200)


# In[ ]:


gb_mod.fit(X_read_train, y_read_yn_train)


# In[ ]:


y_read_preds_gb = gb_mod.predict_proba(X_read_test)[:,1]


# In[ ]:


y_read_preds_gb_act = gb_mod.predict(X_read_test)


# In[ ]:


read_gb_TPR, read_gb_FPR, read_gb_thresholds = roc_curve(y_read_yn_test, y_read_preds_gb, pos_label=None, sample_weight=None, drop_intermediate=True)


# In[ ]:


plotroc(read_gb_TPR, read_gb_FPR)


# In[ ]:


print(classification_report(y_read_yn_test, y_read_preds_gb_act))


# In[ ]:


## Hypothesis: Of discharged patients, most readmissions happen in the first 10 days. If this is true, how does it impact
## our ability to use time cutoffs to groups patients into opt-ins or opt-outs?


# In[ ]:


# fighist_TOC = plt.figure(figsize=(12,6))
# ax1 = fighist_TOC.add_subplot(111)
# ax1.set_title('Histogram, Discharge to Next Admission, days (TOC)')
# ax1.hist(np.array(progs[progs['PRGM_NM']=='Transitions of Care - Post Discharge']['index_to_next_days'].dropna()), bins = 150, alpha = 0.8, density=1, label='TOC')
# ax1.set_ylabel('Percent of Patients')
# ax1.set_xlabel('days to next admission')
# ax1.set_xlim(left=0, right=200)
# ax1.legend();


# In[ ]:



progs_toc_w_lace.pivot_table(['preds', 'is_30_TOC_adm'], index='RGON_NM', aggfunc='mean')


# # CHF

# In[ ]:


progs_chf = progs


# In[ ]:


progs_chf = progs_chf[progs_chf['PRGM_NM']=='DM - HF'].reset_index(drop=True)


# In[ ]:


## create assessments table for CHF

assess_chf_mhfk = assess[assess['ASES_NM']=='MCG - Heart Failure - Knowledge of Condition and Treatment Plan'].drop_duplicates(subset='PTNT_ASES_DK')
assess_chf_mdhf = assess[assess['ASES_NM']=='MCG - DMG - Heart Failure'].drop_duplicates(subset='PTNT_ASES_DK')
assess_chf_mdhfk = assess[assess['ASES_NM']=='MCG - DMG - Heart Failure - Knowledge of Condition and Treatment Plan'].drop_duplicates(subset='PTNT_ASES_DK')
assess_chf_patout = assess[(assess['ASES_NM']=='Patient Outreach Encounter') & (assess['ANSR_TXT']=='Successful')]
frames = [assess_chf_mhfk, assess_chf_mdhf, assess_chf_mdhfk, assess_chf_patout]
assess_chf = pd.concat(frames)


# In[ ]:


assess_chf = assess_chf.reset_index()


# In[ ]:


## cnt chf related assessments during programs
cnt_chf_90 = find_assessments_during_program_chf(progs_chf, assess_chf)


# In[ ]:


progs_chf['chf_ases_dur_chf'] = cnt_chf_90
cm_chf_ases_yn = adms_to_one_zero_v2(cnt_chf_90)


# In[ ]:


## ADD ALOS Data & Readmissions


# In[ ]:


progs_chf.to_csv('data/progs_chf.csv')


# In[ ]:


## Initial numbers for HF Difference of Differences


# ## COPD

# In[ ]:


progs_copd = progs


# In[ ]:


progs_copd.columns


# In[ ]:


progs_copd = progs_copd[progs_copd['PRGM_NM']=='DM - CLD'].reset_index(drop=True)


# In[ ]:


rel_empi = list(progs_copd['EMPI'].unique())
rel_admits = admits[admits['EMPI'].isin(rel_empi)].reset_index(drop=True)


# In[ ]:


def tag_copd_admissions(dm_progs, rel_admits):
    dm_admit_yn = list(np.zeros(rel_admits.shape[0]))
    dm_optin_yn = np.empty(rel_admits.shape[0])
    dm_optin_yn[:] = np.nan
    dm_optin_yn = list(dm_optin_yn)
    dm_week = np.empty(rel_admits.shape[0])
    dm_week[:] = np.nan
    dm_week = list(dm_optin_yn)
    for index, row in dm_progs.iterrows():
        for index2, row2 in rel_admits.iterrows():
            if (row2['EMPI']==row['EMPI']) & (row2['admit_date'] > (row['prog_create_date']-timedelta(days=90))) & (row2['admit_date'] < (row['prog_create_date']+timedelta(days=90))):
                dm_admit_yn[index] = 1
                dm_optin_yn[index] = (row['chf_ases_dur_copd']>0)
                dm_week[index] = int(((row2['admit_date']-row['prog_create_date'])/timedelta(days=1))/7)
    return dm_admit_yn, dm_optin_yn, dm_week


# In[ ]:


copd_admit_yn, copd_optin_yn, copd_week = tag_copd_admissions(progs_copd, rel_admits)


# In[ ]:


rel_admits['copd_admit_yn'] = copd_admit_yn
rel_admits['copd_optin_yn'] = copd_optin_yn
rel_admits['copd_optin_week'] = copd_week


# In[ ]:


rel_admits.columns


# In[ ]:


rel_admits.pivot_table('ALOS_N', index='copd_optin_week', columns='copd_optin_yn', aggfunc=['sum'])


# #### Need these from Emmy

# In[ ]:


## create assessments table for COPD

assess_copd_mcopdk = assess[assess['ASES_NM']=='MCG - Chronic Obstructive Pulmonary Disease - Knowledge of Condition and Treatment Plan'].drop_duplicates(subset='PTNT_ASES_DK')
assess_copd_mdchr = assess[assess['ASES_NM']=='MCG - DMG - Chronic Lung Disease (CLD)'].drop_duplicates(subset='PTNT_ASES_DK')
assess_copd_mdcld = assess[assess['ASES_NM']=='MCG - DMG - CLD'].drop_duplicates(subset='PTNT_ASES_DK')
assess_copd_patout = assess[(assess['ASES_NM']=='Patient Outreach Encounter') & (assess['ANSR_TXT']=='Successful')]
frames = [assess_copd_mcopdk, assess_copd_mdchr, assess_copd_mdcld, assess_copd_patout]
assess_copd = pd.concat(frames)


# In[ ]:


assess_copd = assess_copd.reset_index()


# In[ ]:


## cnt chf related assessments during programs
cnt_copd_90 = find_assessments_during_program_chf(progs_copd, assess_copd)


# In[ ]:


progs_copd['chf_ases_dur_copd'] = cnt_copd_90
cm_copd_ases_yn = adms_to_one_zero_v2(cnt_copd_90)


# In[ ]:


## ADD ALOS Data & Readmissions


# In[ ]:


progs_copd.to_csv('data/progs_copd.csv')


# ## Write to file

# In[ ]:


progs.to_csv('data/progs_clean.csv')


# ## Assessments to Program Matching

# In[ ]:


progs.head()


# In[ ]:


# assess_all = pd.read_csv('data/assess_all.rpt', sep='|')

# assess_all = assess_all[assess_all['ASES_DT']!='2917-12-26 00:00:00.000']

# assess_all['ASES_DT'] = pd.to_datetime(assess_all['ASES_DT'])
# assess_all['EFF_FM_TS'] = pd.to_datetime(assess_all['EFF_FM_TS'])

# list_o_ases = list(assess_all['ASES_NM'].unique())

# progs_hf = progs[progs['PRGM_NM']=='DM - HF']

# progs_cld = progs[progs['PRGM_NM']=='DM - CLD']

# assess_counts_toc = find_assessments_during_program_all(progs_toc, assess_all, keep_ases)

# assess_counts_hf = find_assessments_during_program_all(progs_hf, assess_all, keep_ases)

# assess_counts_cld = find_assessments_during_program_all(progs_cld, assess_all, keep_ases)

# counts_of_assesses = np.column_stack((assess_counts_toc, assess_counts_hf, assess_counts_cld))

# assessments_by_prog = pd.DataFrame(counts_of_assesses, keep_ases, ['TOC', 'HF', 'CLD'])

# assessments_by_prog.to_csv('data/assessments_by_prog.csv')

# counts = count_ases(assess_all)

# keep_ases = []
# for num in range(194):
#     if counts[num]>100:
#         keep_ases.append(list_o_ases[num])

