
# coding: utf-8

# Instructions for how to rerun this analysis.
# 1. Update 3 queries - progs, assessments, and admits, they are stored in github.com/tysonjens
# 2. Run queries and save locally to your computer
# 3. Update "read_csv" commands below to corresopnd to your local paths
# 4. Run the script and note outputs for the several analyses

# In[198]:


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

# In[248]:


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


# In[251]:


## function that finds the first assessment completed after discharge and records the date
def find_first_assess(programs, assessments):
    assessments = assessments.sort_values(by='ASES_DT', ascending=True)
    first_assess_dates = np.empty(programs.shape[0])
    first_assess_dates[:] = np.nan
    first_assess_dates = list(first_assess_dates)
    first_assess_name = np.empty(programs.shape[0])
    first_assess_name[:] = np.nan
    first_assess_name = list(first_assess_name)
    for index, row in programs.iterrows():
        assess_pat = assessments[assessments['EMPI']==row['EMPI']]
        for index2, row2 in assess_pat.iterrows():
            if row2['ASES_DT'] > row['index_date']:
                first_assess_dates[index] = row2['ASES_DT']
                first_assess_name[index] = row2['ASES_NM']
                break
            else:
                continue
    return first_assess_dates, first_assess_name


# In[203]:


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


# In[204]:


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


# In[205]:


def find_assessments_during_program(programs, assess):
    assess_cnt = np.zeros(programs.shape[0])
    assess_cnt = list(assess_cnt)
    for index, row in programs.iterrows():
        assess_pat = assess[(assess['EMPI']==row['EMPI']) &
                             (assess['ASES_DT'] > (row['prog_create_date']-timedelta(days=10))) &
                             (assess['ASES_DT'] < (row['prog_create_date']+timedelta(days=35)))]
        assess_cnt[index] = assess_pat.shape[0]
    return assess_cnt


# In[206]:


def find_assessments_during_program_chf(programs, assess):
    assess_cnt = np.zeros(programs.shape[0])
    assess_cnt = list(assess_cnt)
    for index, row in programs.iterrows():
        assess_pat = assess[(assess['EMPI']==row['EMPI']) &
                             (assess['ASES_DT'] > (row['prog_create_date']-timedelta(days=1))) &
                             (assess['ASES_DT'] < (row['prog_create_date']+timedelta(days=90)))]
        assess_cnt[index] = assess_pat.shape[0]
    return assess_cnt


# In[207]:


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


# In[208]:


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


# In[209]:


def adms_to_one_zero_v2(ases):
    ases_yn = []
    for num in ases:
        if num>0:
            ases_yn.append(1)
        else:
            ases_yn.append(0)
    return ases_yn


# In[210]:


def to_one_zero(probas, value):
    ones_zeros = []
    for num in probas:
        if num>value:
            ones_zeros.append(1)
        else:
            ones_zeros.append(0)
    return np.array(ones_zeros)


# In[211]:


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


# In[212]:


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
                beddays += row2['length_of_stay']
        admits_in_window[index] = count
        beddays_in_window[index] = beddays
    return admits_in_window, beddays_in_window


# In[213]:


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
                beddays += row2['length_of_stay']
        admits_in_window[index] = count
        beddays_in_window[index] = beddays
    return admits_in_window, beddays_in_window


# In[214]:


## function that counts the number of admits in a window AFTER the index readmission
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


# In[215]:


def adms_to_one_zero(progs, field):
    adm_yn = []
    for index, row in progs.iterrows():
        if row[field]>0:
            adm_yn.append(1)
        else:
            adm_yn.append(0)
    return adm_yn


# In[216]:


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


# In[217]:


def bootstrap_ci_coefficients(X_train, y_train, num_bootstraps, class_weight={0: .5, 1: 2}):
    bootstrap_estimates = []
    for i in np.arange(num_bootstraps):
        sample_index = np.random.choice(range(0, len(y_train)), len(y_train))
        X_samples = X_train[sample_index]
        y_samples = y_train[sample_index]
        lm = linear_model.LogisticRegression(class_weight=class_weight, penalty='l1', C=.1)
        lm.fit(X_samples, y_samples)
        bootstrap_estimates.append(lm.coef_[0])
    bootstrap_estimates = np.asarray(bootstrap_estimates)
    return bootstrap_estimates


# In[218]:


def bootstrap_ci_coefficients_lin(X_train, y_train, num_bootstraps):
    bootstrap_estimates = []
    for i in np.arange(num_bootstraps):
        sample_index = np.random.choice(range(0, len(y_train)), len(y_train))
        X_samples = X_train[sample_index]
        y_samples = y_train[sample_index]
        lm = linear_model.LogisticRegression(penalty='l1', C=.1, class_weight=class_weight)
        lm.fit(X_samples, y_samples)
        bootstrap_estimates.append(lm.coef_)
    bootstrap_estimates = np.asarray(bootstrap_estimates)
    return bootstrap_estimates


# In[219]:


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
                dm_optin_yn[index] = (row['copd_ases_dur_copd']>0)
                dm_week[index] = int(((row2['admit_date']-row['prog_create_date'])/timedelta(days=1))/7)
    return dm_admit_yn, dm_optin_yn, dm_week


# In[220]:


def tag_chf_admissions(dm_progs, rel_admits):
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
                dm_optin_yn[index] = (row['chf_ases_dur_chf']>0)
                dm_week[index] = int(((row2['admit_date']-row['prog_create_date'])/timedelta(days=1))/7)
    return dm_admit_yn, dm_optin_yn, dm_week


# ## Read in Data

# In[221]:


## READ In Programs, Admissions, and Lace data
## Could have script run this
progs = pd.read_csv('data/20181025progs.rpt', sep='|', low_memory=False)


# In[222]:


admits = pd.read_csv('data/20181105admissions.csv', sep='|', low_memory=False)


# In[223]:


assess = pd.read_csv('data/20181025assessments.csv.sql', sep='|', low_memory=False)


# To address the challenge of using the correct TOC assessments, we have isolated them as part of the query (instead of handling in Python). assess_toc1 has 'TOC Post Discharge Outreach' and 'Patient Outreach Encounter'.  Assess_toc2 has "MCG - DMG - Post hospitalization follow up' and 'Post Hospitalization Follow Up'

# In[224]:


##assess_toc1 = pd.read_csv()


# In[225]:


assess_toc2 = pd.read_csv('data/20181105assessments_toc2.csv', sep='|', low_memory=False)


# ## Clean Data

# #### Clean progs

# In[226]:


## convert dates to date time
progs['prog_create_date'] = pd.to_datetime(progs['ASGN_TMS'])
progs['date_of_birth'] = pd.to_datetime(progs['DOB'])
progs['prog_end_date'] = pd.to_datetime(progs['END_TMS'])


# In[227]:


## create opt-in and opt-out with stop reason
opt_in = {'Admit to Facility':1, 'Discharge to Custodial':1, 'Discharged to Hospice':1, 
          'Discharged to PCP':1, 'Goals Met':1, 'Referred to other program':1, 
          'Refused':0, 'Unable to Reach':0, 'Barriers to participation':0, 'Expired':np.nan, 
         'Criteria not met':np.nan, 'Disenrolled from HP/Medical Group':np.nan, '6 months/1 year post transplant':np.nan,
       'Pharmacy - physician recommended':np.nan}
progs['is_optin'] = progs['PRGM_STOP_RSN'].replace(opt_in)


# In[228]:


## replace date_of_birth with age
progs['age'] = datetime.now() - progs['date_of_birth']
progs['age'] = progs['age'] / timedelta(days=1) / 365
progs['age'].fillna(float(progs['age'].mean()), inplace=True)


# In[229]:


new_sex = {'F':0, 'M':1}
progs['is_male'] = progs['Sex'].replace(new_sex)
progs['is_male'].fillna(0, inplace=True)


# In[230]:


## drop unneeded columns
prog_cols_drop = ['PTNT_DK', 'DOB', 'Sex', 'ASGN_TMS', 'END_TMS', 'TNT_MKT_BK', 'date_of_birth', 'PRGM_STS']
progs = progs.drop(prog_cols_drop, axis=1)


# In[231]:


progs = progs[(progs['prog_create_date']>'2018-04-01') & (progs['prog_create_date']<'2018-10-29')]


# In[232]:


progs['EMPI'] = progs['EMPI'].astype(int)


# In[233]:


progs = progs.reset_index()


# In[234]:


## convert object columns to categoricals
for col in ['RGON_NM', 'HP_NM', 'LOB_SUB_CGY', 'PRGM_NM', 'PRGM_STOP_RSN']:
    progs[col] = progs[col].astype('category')


# #### Clean Admits

# In[235]:


admits['EMPI'].fillna(999999999, inplace=True)

admits['EMPI'] = admits['EMPI'].astype(int)


# In[237]:


## convert dates to date time
admits['admit_date'] = pd.to_datetime(admits['ACT_ADM_DT'])
admits['discharge_date'] = pd.to_datetime(admits['ACT_DISCH_DT'])


# In[238]:


## calculate LOS using admit date and discharge date
admits['length_of_stay'] = ((admits['discharge_date']-admits['admit_date']) / np.timedelta64(1, 'D'))


# In[239]:


# ## This portion was used with Michael Maley's admission query. May not be needed anymore.
# ## replace Model with is_group
# new_Model = {'GROUP':1, 'IPA':0, 'JOINT VENTURE':0, 'OPEN ACCESS':0, 'OTHER':0}
# admits['is_group'] = admits['Model'].replace(new_Model)
# admits['is_group'].fillna(0, inplace=True)


# In[240]:


## drop unneeded columns
admits_cols_drop = ['REFERRAL_KEY', 'PATIENT', 'ACT_ADM_DT', 'ACT_DISCH_DT']
admits = admits.drop(admits_cols_drop, axis=1)


# In[241]:


admits_all = admits


# In[242]:


# ## This portion was used with Michael Maley's admission query. May not be needed anymore.
# ## only keep acute admissions
# admits = admits[admits['Acuity']=='ACUTE']


# In[243]:


# ## This was required for Michael Maley's query
# ## change objects to categories
# cols_to_category = ['REGION', 'SITE', 'LOB', 'Acuity', 'Facility',
#        'RefType', 'DayType', 'AdmissionType', 'DischDx1', 'DischDx1Desc', 'SurgeryPx', 'DISPOSITION', 'ACSA_CAT']
# for col in cols_to_category:
#     admits[col] = admits[col].astype('category')


# #### Clean Assessments

# In[244]:


## REmove assessments with weird filler date '2917-12-26 00:00:00.000'
assess = assess[assess['ASES_DT']!='2917-12-26 00:00:00.000']
#assess_toc1 = assess_toc1[assess_toc1['ASES_DT']!='2917-12-26 00:00:00.000']
assess_toc2 = assess_toc2[assess_toc2['ASES_DT']!='2917-12-26 00:00:00.000']


# In[245]:


assess['ASES_DT'] = pd.to_datetime(assess['ASES_DT'])
#assess_toc1['ASES_DT'] = pd.to_datetime(assess_toc1['ASES_DT'])
assess_toc2['ASES_DT'] = pd.to_datetime(assess_toc2['ASES_DT'])
#assess['EFF_FM_TS'] = pd.to_datetime(assess['EFF_FM_TS'])


# In[246]:


# new_text = {'Successful (enter number you\'re calling)':'Success'}
# assess['ANSR_TXT'] = assess['ANSR_TXT'].replace(new_text)


# ## Feature Engineering

# In[252]:


index_dates = find_index_admit(progs, admits_all)
progs['index_date'] = index_dates
progs['index_date'] = pd.to_datetime(progs['index_date'])


# In[253]:


first_admissions = find_first_admission_after_enroll(progs, admits)
progs['frst_adm_aftr_enrl'] = first_admissions
progs['frst_adm_aftr_enrl'] = pd.to_datetime(progs['frst_adm_aftr_enrl'])


# In[254]:


LACE = assess[assess['ASES_NM']=='LACE']


# In[255]:


lace_scores = find_lace_prior_to_enroll(progs, LACE)
progs['lace_score'] = lace_scores


# In[256]:


## create count and y/n columns
cm_toc_ases = find_assessments_during_program(progs, assess_toc2)
cm_toc_ases2 = adms_to_one_zero_v2(cm_toc_ases)
progs['cnt_toc_cm_touch'] = cm_toc_ases
progs['toc_cm_touch_yn'] = cm_toc_ases2


# In[257]:


## calc number of admits that occur within 30 day window after program begins
thirty_day_after_adm, thirty_day_after_bd  = get_adm_after(progs, admits, window_size=30)
progs['adm_30_after'] = thirty_day_after_adm
progs['bd_30_after'] = thirty_day_after_bd


# In[258]:


## calc number of admits that occur within 90 day window after program begins
ninety_day_after_adm, ninety_day_after_bd = get_adm_after(progs, admits, window_size=90)
progs['adm_90_after'] = ninety_day_after_adm
progs['bd_90_after'] = ninety_day_after_bd


# In[260]:


thirty_day_after_TOC = get_adm_after_TOC(progs, admits)
progs['adm_30_after_TOC'] = thirty_day_after_TOC


# In[262]:


## calc number of admits that occur within 90 day window before program begins
ninety_day_before_adm , ninety_day_before_bd = get_adm_before(progs, admits)
progs['adm_90_before'] = ninety_day_before_adm
progs['bd_90_before'] = ninety_day_before_bd


# In[263]:


progs['time_to_enroll'] = progs['prog_create_date']-progs['index_date']
progs['time_to_enroll'] = progs['time_to_enroll']/ timedelta(days=1)


# In[264]:


progs['prog_duration'] = progs['prog_end_date']-progs['prog_create_date']
progs['prog_duration'] = progs['prog_duration']/ timedelta(days=1)


# In[265]:


progs['index_to_next_days'] = progs['frst_adm_aftr_enrl']-progs['index_date']
progs['index_to_next_days'] = progs['index_to_next_days']/ timedelta(days=1)


# In[266]:


adm_yn = adms_to_one_zero(progs, 'adm_30_after_TOC')


# In[267]:


progs['is_30_TOC_adm'] = adm_yn


# In[268]:


readm_yn = adms_to_one_zero(progs, 'readm_30_after_TOC')


# In[269]:


progs['is_30_TOC_readm'] = readm_yn


# ## Exploratory Data Analysis

# In[171]:


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


# In[172]:


# fighist = plt.figure(figsize=(12,6))
# ax1 = fighist.add_subplot(111)
# ax1.set_title('Histogram, time to enroll post discharge')
# #ax1.hist(np.array(progs[progs['PRGM_NM']=='Transitions of Care - Post Discharge']['time_to_enroll'].dropna()), bins = 100, alpha = 0.4, density=1, label='TOC')
# ax1.hist(np.array(progs[progs['PRGM_NM']=='DM - CLD']['time_to_enroll'].dropna()), bins = 40, alpha = 0.6, density=1, label='CLD')
# ax1.hist(np.array(progs[progs['PRGM_NM']=='DM - HF']['time_to_enroll'].dropna()), bins = 40, alpha = 0.6, density=1, label='HF')
# ax1.set_ylabel('Percent of Patients')
# ax1.set_xlabel('days to enrollment')
# ax1.legend();


# In[173]:


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


# In[174]:


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


# In[175]:


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


# In[176]:


# fighist_TOC = plt.figure(figsize=(12,6))
# ax1 = fighist_TOC.add_subplot(111)
# ax1.set_title('Histogram, time to enroll (TOC)')
# ax1.hist(np.array(progs[progs['PRGM_NM']=='Transitions of Care - Post Discharge']['time_to_enroll'].dropna()), bins = 200, alpha = .6, density=1, color='green', label='TOC')
# ax1.set_ylabel('Percent of Patients')
# ax1.set_xlabel('days to enrollment')
# ax1.set_xlim(left=0, right=40)
# ax1.legend();


# In[177]:


# progs[progs['PRGM_NM']=='Transitions of Care - Post Discharge'].pivot_table(values='adm_30_after_TOC', index='PRGM_STOP_RSN', aggfunc=['count','mean'], dropna=True)


# In[178]:


## Understand relationship between LACE and time to enroll


# In[179]:


# progs[progs['PRGM_NM']=='Transitions of Care - Post Discharge'].pivot_table(values='adm_30_after_TOC', index='lace_score', columns='is_optin', aggfunc=['count','mean'], dropna=True)


# In[180]:


# ## graph violinplots of optin vs. optout LACE Scores.
# fighist_lace_age = plt.figure(figsize=(12,6))
# ax1 = fighist_lace_age.add_subplot(111)
# sns.boxplot(ax=ax1, x="lace_score", y='age', data=progs[progs['PRGM_NM']=='Transitions of Care - Post Discharge'])


# In[181]:


# ## graph violinplots of optin vs. optout LACE Scores.
# sns.violinplot(x="is_optin", y='lace_score', data=progs[progs['PRGM_NM']=='Transitions of Care - Post Discharge'])


# In[182]:


## use logisic regression to test whether opt_in has impact on readmit_30_days_TOC, controlling for other variables.


# In[183]:


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


# ## TOC

# #### Set Up train test split and x matricies

# In[225]:


progs_toc = progs[progs['PRGM_NM']=='Transitions of Care - Post Discharge']


# In[226]:


progs_toc_w_lace = progs_toc[(progs_toc['lace_score'].isna()==False) & (progs_toc['prog_create_date']<'2018-09-29')].reset_index()


# In[227]:


progs_toc_ent = pd.get_dummies(pd.Series(list(progs_toc_w_lace['ENT_TYPE'])))


# In[228]:


progs_toc_lob = pd.get_dummies(pd.Series(list(progs_toc_w_lace['LOB_SUB_CGY'])))


# In[229]:


progs_toc_rgn = pd.get_dummies(pd.Series(list(progs_toc_w_lace['RGON_NM'])))


# In[230]:


progs_toc_w_lace = pd.concat([progs_toc_w_lace, progs_toc_lob, progs_toc_ent, progs_toc_rgn], axis=1)


# In[231]:


X = np.array(progs_toc_w_lace[['toc_cm_touch_yn','age', 'is_male', 'bd_90_before', 'lace_score', 'time_to_enroll',
                               'prog_duration', 'COMMERCIAL', 'MEDI-CAL', 'IPA', 'LA/DOWNTOWN', 'LONG BEACH', 'MAGAN',
                               'ORANGE COUNTY', 'SAN FERNANDO VALLEY', 'SOUTH BAY', 'VILLAGE HEALTH']])


# In[232]:


X1 = np.array(progs_toc_w_lace[['toc_cm_touch_yn','age', 'is_male', 'lace_score']])


# In[233]:


imput = Imputer(strategy='median')


# In[234]:


X = imput.fit_transform(X)


# #### Admissions - Use until we determine best measure of readmissions

# In[235]:


y_ad_yn = np.array(progs_toc_w_lace['is_30_TOC_adm'])


# In[236]:


X_ad_train, X_ad_test, y_ad_yn_train, y_ad_yn_test = train_test_split(X, y_ad_yn, test_size=.15)


# In[237]:


class_weight={0: 0.5, 1: 2}


# In[238]:


model_ad = linear_model.LogisticRegression(C=.1, penalty='l1', class_weight=class_weight)


# In[239]:


model_ad.fit(X, y_ad_yn)


# In[240]:


model_ad.coef_


# In[241]:


model_ad.intercept_


# In[242]:


y_ad_yn_preds = model_ad.predict_proba(X)[:,1]


# In[243]:


y_ad_yn_preds_act = model_ad.predict(X)


# In[244]:


progs_toc_w_lace['is_30_TOC_adm_pred'] = y_ad_yn_preds_act


# In[245]:


ad_yn_bootstraps = bootstrap_ci_coefficients(X, y_ad_yn, 2000)


# In[246]:


ad_yn_bootstraps = pd.DataFrame(ad_yn_bootstraps, columns=['toc_cm_touch_yn','age', 'is_male', 'bd_90_before', 'lace_score', 'time_to_enroll',
                               'prog_duration', 'COMMERCIAL', 'MEDI-CAL', 'IPA', 'LA/DOWNTOWN', 'LONG BEACH', 'MAGAN',
                               'ORANGE COUNTY', 'SAN FERNANDO VALLEY', 'SOUTH BAY', 'VILLAGE HEALTH'])


# In[247]:


fig, axes = plt.subplots(4,4, figsize=(16,16))
col_names = ['toc_cm_touch_yn','age', 'is_male', 'bd_90_before', 'lace_score', 'time_to_enroll',
                               'prog_duration', 'COMMERCIAL', 'MEDI-CAL', 'IPA', 'LA/DOWNTOWN', 'LONG BEACH', 'MAGAN',
                               'ORANGE COUNTY', 'SAN FERNANDO VALLEY', 'SOUTH BAY', 'VILLAGE HEALTH']

for m, ax in zip(col_names, axes.flatten()):
    ax.hist(ad_yn_bootstraps[m], bins=50)
    ax.set_title(m)


# In[248]:


np.exp(ad_yn_bootstraps.mean(axis=0))-1


# In[249]:


ad_coefs = model_ad.coef_


# In[250]:


pd.concat([pd.Series(['toc_cm_touch_yn','age', 'is_male', 'bd_90_before', 'lace_score', 'time_to_enroll',
                               'prog_duration', 'COMMERCIAL', 'MEDI-CAL', 'IPA', 'LA/DOWNTOWN', 'LONG BEACH', 'MAGAN',
                               'ORANGE COUNTY', 'SAN FERNANDO VALLEY', 'SOUTH BAY', 'VILLAGE HEALTH']), pd.Series(np.exp(ad_coefs)[0])-1], axis=1, )


# In[251]:


ad_TPR, ad_FPR, ad_thresholds = roc_curve(y_ad_yn, y_ad_yn_preds, pos_label=None, sample_weight=None, drop_intermediate=True)


# In[252]:


plotroc(ad_TPR, ad_FPR)


# In[253]:


progs_toc_w_lace['prog_create_date'].max()


# In[254]:


progs_toc_w_lace.to_csv('data/scriptout_progs_toc_w_lace.csv')


# #### 30 Day Bed Days

# In[213]:


y_30_bd = np.array(progs_toc_w_lace['bd_30_after'])


# In[214]:


X_bd_train, X_bd_test, y_bd_train, y_bd_test = train_test_split(X, y_30_bd, test_size=.25)


# In[215]:


y_30_bd.mean()


# In[216]:


model_bd = linear_model.LinearRegression()


# In[217]:


model_bd.fit(X_bd_train, y_bd_train)


# In[218]:


bd_coefs = model_bd.coef_


# In[219]:


y_bd_preds = model_bd.predict(X_bd_test)


# In[220]:


y_bd_test.mean()


# In[221]:


y_bd_preds.mean()


# In[222]:


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


bd_bootstraps.mean(axis=0)


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


assess['ASES_NM'].value_counts()


# In[ ]:


progs_toc_w_lace['is_30_TOC_adm'].mean()


# In[ ]:


gb_mod = GradientBoostingClassifier(learning_rate=0.05, max_depth=3, n_estimators=150)


# In[ ]:


gb_mod.fit(X_ad_train, y_ad_yn_train)


# In[ ]:


y_ad_preds_gb = gb_mod.predict_proba(X_ad_test)[:,1]


# In[ ]:


y_ad_preds_gb_act = gb_mod.predict(X_ad_test)


# In[ ]:


ad_gb_TPR, ad_gb_FPR, ad_gb_thresholds = roc_curve(y_ad_yn_test, y_ad_preds_gb, pos_label=None, sample_weight=None, drop_intermediate=True)


# In[ ]:


plotroc(ad_gb_TPR, ad_gb_FPR)


# In[ ]:


print(classification_report(y_ad_yn_test, y_ad_preds_gb_act))


# #### TOC - Use model to identify segments where we're doing better, worse than average

# In[ ]:


gb_mod = GradientBoostingClassifier()


# In[ ]:


gb_mod.fit(X, y_ad_yn)


# In[ ]:


progs_toc_w_lace.columns


# In[ ]:


progs_toc_w_lace.pivot_table(values="EMPI", index='is_30_TOC_adm', columns='toc_cm_touch_yn',
                            aggfunc='count')


# In[ ]:


progs_toc_w_lace.to_csv('data/prog_toc_lace.csv')


# In[ ]:


y_gb_preds_act = to_one_zero(y_ad_preds_gb_full, .355)


# In[ ]:


y_gb_preds_act.mean()


# In[ ]:


y_ad_preds_gb_full = gb_mod.predict_proba(X)[:,1]


# In[ ]:


y_ad_preds_gb_act_full = gb_mod.predict(X)


# In[ ]:


ad_gb_full_TPR, ad_gb_full_FPR, ad_gb_full_thresholds = roc_curve(y_ad_yn_test, y_ad_preds_gb, pos_label=None, sample_weight=None, drop_intermediate=True)


# In[ ]:


plotroc(ad_gb_full_TPR, ad_gb_full_FPR)


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


progs_toc_w_lace['gb_ad_preds'] = y_gb_preds_act


# In[ ]:


progs_toc_w_lace['is_30_TOC_adm'].mean()


# In[ ]:


len(y_ad_preds_gb_act_full)


# In[ ]:


progs_toc_w_lace.pivot_table(['gb_ad_preds', 'is_30_TOC_adm'], index='RGON_NM', aggfunc=['mean', 'count']).to_csv('data/subgroup_reg.csv')


# In[ ]:


progs_toc_w_lace.pivot_table(['gb_ad_preds', 'is_30_TOC_adm'], index='LOB_SUB_CGY', aggfunc=['mean', 'count']).to_csv('data/subgroup_lob.csv')


# In[ ]:


progs_toc_w_lace.pivot_table(['gb_ad_preds', 'is_30_TOC_adm'], index='ENT_TYPE', aggfunc=['mean', 'count']).to_csv('data/subgroup_ent.csv')


# In[ ]:


progs_toc_w_lace.pivot_table(['gb_ad_preds', 'is_30_TOC_adm'], index='ASGN_USR', aggfunc=['mean', 'count']).to_csv('data/subgroup_user.csv')


# In[ ]:


progs_toc_w_lace.pivot_table(['gb_ad_preds', 'is_30_TOC_adm'], index='CLNC_NM', aggfunc=['mean', 'count']).to_csv('data/subgroup_site.csv')


# In[ ]:


print(confusion_matrix(progs_toc_w_lace['is_30_TOC_adm'], progs_toc_w_lace['gb_ad_preds']))


# In[ ]:


(3130+325)/(3130+325+181+181)


# In[ ]:


progs_toc_w_lace.columns


# # CHF

# In[ ]:


progs_chf = progs


# In[ ]:


progs_chf = progs_chf[(progs_chf['PRGM_NM']=='DM - HF') & (progs_chf['prog_create_date']<'2018-6-15')].reset_index(drop=True)


# In[ ]:


progs[progs['EMPI']==1001603330]


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


# #### Parallel Assumption Testing

# In[ ]:


rel_empi_chf = list(progs_chf['EMPI'].unique())
rel_admits_chf = admits[admits['EMPI'].isin(rel_empi_chf)].reset_index(drop=True)


# In[ ]:


chf_admit_yn, chf_optin_yn, chf_week = tag_chf_admissions(progs_chf, rel_admits_chf)


# In[ ]:


rel_admits_chf['chf_admit_yn'] = chf_admit_yn
rel_admits_chf['chf_optin_yn'] = chf_optin_yn
rel_admits_chf['chf_optin_week'] = chf_week


# In[ ]:


rel_admits_chf.pivot_table('ALOS_N', index='chf_optin_week', columns='chf_optin_yn', aggfunc='sum').to_csv('data/chf_bd.csv')


# In[ ]:


rel_admits_chf.pivot_table('ALOS_N', index='chf_optin_week', columns='chf_optin_yn', aggfunc='sum')


# In[ ]:


rel_admits_chf.pivot_table('ADMIT_CT', index='chf_optin_week', columns='chf_optin_yn', aggfunc='sum').to_csv('data/chf_adm.csv')


# In[ ]:


progs_copd.columns


# ## COPD

# In[ ]:


progs_copd = progs


# In[ ]:


progs_copd = progs_copd[(progs_copd['PRGM_NM']=='DM - CLD') & (progs_copd['prog_create_date']<'2018-6-15')].reset_index(drop=True)


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


progs_copd['copd_ases_dur_copd'] = cnt_copd_90
cm_copd_ases_yn = adms_to_one_zero_v2(cnt_copd_90)


# #### Parallel Assumption Testing

# In[ ]:


rel_empi = list(progs_copd['EMPI'].unique())
rel_admits = admits[admits['EMPI'].isin(rel_empi)].reset_index(drop=True)


# In[ ]:


copd_admit_yn, copd_optin_yn, copd_week = tag_copd_admissions(progs_copd, rel_admits)


# In[ ]:


rel_admits['copd_admit_yn'] = copd_admit_yn
rel_admits['copd_optin_yn'] = copd_optin_yn
rel_admits['copd_optin_week'] = copd_week


# In[ ]:


progs_copd.count()


# In[ ]:


rel_admits.pivot_table('ALOS_N', index='copd_optin_week', columns='copd_optin_yn', aggfunc='sum').to_csv('data/copd_bd.csv')


# In[ ]:


rel_admits.pivot_table('ADMIT_CT', index='copd_optin_week', columns='copd_optin_yn', aggfunc='sum').to_csv('data/copd_adm.csv')


# In[ ]:


progs_copd.columns


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


# In[ ]:


import scipy.stats as st


# In[ ]:


st.binom.cdf(5, 10, .15)


# In[ ]:


progs_toc_w_lace.columns


# ## Lace only model

# In[ ]:


X_lace = np.array(progs_toc_w_lace[['lace_score']])


# In[ ]:


y_read_yn = np.array(progs_toc_w_lace['is_30_TOC_adm'])


# In[ ]:


X_read_train, X_read_test, y_read_yn_train, y_read_yn_test = train_test_split(X_lace, y_read_yn, test_size=.2)


# In[ ]:


model_ad = linear_model.LogisticRegression(C=1, penalty='l1', class_weight=class_weight)


# In[ ]:


model_ad.fit(X_read_train, y_read_yn_train)


# In[ ]:


model_ad.coef_


# In[ ]:


y_read_yn_preds = model_ad.predict_proba(X_read_test)[:,1]


# In[ ]:


y_read_yn_preds_act = model_ad.predict(X_read_test)


# In[ ]:


y_read_yn_test.mean()


# In[ ]:


y_ad_yn_preds_act.mean()


# In[ ]:


ad_gb_full_TPR, ad_gb_full_FPR, ad_gb_full_thresholds = roc_curve(y_read_yn_test, y_read_yn_preds, pos_label=None, sample_weight=None, drop_intermediate=True)


# In[ ]:


plotroc(ad_gb_full_TPR, ad_gb_full_FPR)

