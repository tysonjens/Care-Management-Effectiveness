
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dateutil.parser import parse
from pandas.tseries.offsets import Day
# import warnings; warnings.simplefilter('ignore')
pd.set_option("display.max_columns", 100)
pd.set_option('display.max_rows', 500)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


## READ In Programs, Admissions, and Lace data
## Could have script run this
progs = pd.read_csv('data/Program_Patdim.csv')


# In[ ]:


admits = pd.read_csv('data/admissions.csv', sep='|', low_memory=False)


# In[ ]:


## Cleaning, field removal, and data type changes


# In[ ]:


## convert dates to date time
progs['prog_create_date'] = pd.to_datetime(progs['CRT_TMS'])
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


admits['EMPI'].fillna(999999999, inplace=True)


# In[ ]:


admits['EMPI'] = admits['EMPI'].astype(int)


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
prog_cols_drop = ['PTNT_DK', 'PTNT_DK.1', 'DOB', 'Sex', 'CRT_TMS', 'END_TMS', 'TNT_MKT_BK', 'date_of_birth', 'PRGM_STS']
progs = progs.drop(prog_cols_drop, axis=1)


# In[ ]:


progs = progs[(progs['prog_create_date']>'2018-04-01') & (progs['prog_create_date']<'2018-08-01')]


# In[ ]:


progs = progs.reset_index()


# In[ ]:


## convert object columns to categoricals
for col in ['RGON_NM', 'HP_NM', 'LOB_SUB_CGY', 'PRGM_NM', 'PRGM_STOP_RSN']:
    progs[col] = progs[col].astype('category')


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


## change objects to categories
cols_to_category = ['REGION', 'SITE', 'LOB', 'Acuity', 'Facility',
       'RefType', 'DayType', 'AdmissionType', 'DischDx1', 'DischDx1Desc', 'SurgeryPx', 'DISPOSITION', 'ACSA_CAT']
for col in cols_to_category:
    admits[col] = admits[col].astype('category')


# In[ ]:


## create new columns using functions and the admits data

admits = admits.sort_values(by='discharge_date', ascending=False)


# In[ ]:


admits_all = admits


# In[ ]:


## only keep acute admissions
admits = admits[admits['Acuity']=='ACUTE']


# In[ ]:


## function that finds the most recent discharge before a program begins
def find_index_admit(programs, admissions):
    index_dates = list(np.zeros(programs.shape[0]))
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


index_dates = find_index_admit(progs, admits_all)


# In[ ]:


progs['index_date'] = index_dates


# In[ ]:


progs['index_date'].replace(0, np.nan, inplace=True)


# In[ ]:


progs['index_date'] = pd.to_datetime(progs['index_date'])


# In[ ]:


## function that counts the number of admits in a window AFTER a program begins
def get_adm_after(programs, admissions, window_size=30):
    admits_in_window = list(np.zeros(programs.shape[0]))
    for index, row in programs.iterrows():
        admit_pat = admissions[admissions['EMPI']==row['EMPI']]
        count = 0
        for index2, row2 in admit_pat.iterrows():
            if (row2['admit_date'] < (row['prog_create_date']+timedelta(days=window_size))) & (row2['admit_date'] > row['prog_create_date']):
                count+=1
        admits_in_window[index] = count
    return admits_in_window


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


## function that counts the number of admits in a window BEFORE a program begins
def get_adm_before(programs, admissions, window_size=30):
    admits_in_window = list(np.zeros(programs.shape[0]))
    for index, row in programs.iterrows():
        admit_pat = admissions[admissions['EMPI']==row['EMPI']]
        count = 0
        for index2, row2 in admit_pat.iterrows():
            if (row2['admit_date'] > (row['prog_create_date']-timedelta(days=window_size))) & (row2['admit_date'] < row['prog_create_date']):
                count+=1
        admits_in_window[index] = count
    return admits_in_window


# In[ ]:


## calc number of admits that occur within 30 day window after program begins
thirty_day_after = get_adm_after(progs, admits)
progs['adm_30_after'] = thirty_day_after


# In[ ]:


## calc number of admits that occur within 30 day window before program begins
thirty_day_before = get_adm_before(progs, admits)
progs['adm_30_before'] = thirty_day_before


# In[ ]:


progs['time_to_enroll'] = progs['prog_create_date']-progs['index_date']
progs['time_to_enroll'] = progs['time_to_enroll']/ timedelta(days=1)


# In[ ]:


progs['prog_duration'] = progs['prog_end_date']-progs['prog_create_date']
progs['prog_duration'] = progs['prog_duration']/ timedelta(days=1)


# In[ ]:


thirty_day_after_TOC = get_adm_after_TOC(progs, admits)
progs['adm_30_after_TOC'] = thirty_day_after_TOC


# In[ ]:


fighist = plt.figure(figsize=(12,6))
ax1 = fighist.add_subplot(111)
ax1.set_title('Histogram, Days Program Duration')
#ax1.hist(np.array(progs[progs['PRGM_NM']=='Transitions of Care - Post Discharge']['time_to_enroll'].dropna()), bins = 100, alpha = 0.4, density=1, label='TOC')
ax1.hist(np.array(progs[progs['PRGM_NM']=='DM - CLD']['prog_duration'].dropna()), bins = 150, alpha = 0.3, density=1, label='CLD')
ax1.hist(np.array(progs[progs['PRGM_NM']=='DM - HF']['prog_duration'].dropna()), bins = 150, alpha = 0.3, density=1, label='HF')
ax1.set_ylabel('Percent of Patients')
ax1.set_xlabel('Days in Program')
ax1.set_xlim(left=-5, right=130)
ax1.legend();


# In[ ]:


fighist = plt.figure(figsize=(12,6))
ax1 = fighist.add_subplot(111)
ax1.set_title('Histogram, time to enroll post discharge')
#ax1.hist(np.array(progs[progs['PRGM_NM']=='Transitions of Care - Post Discharge']['time_to_enroll'].dropna()), bins = 100, alpha = 0.4, density=1, label='TOC')
ax1.hist(np.array(progs[progs['PRGM_NM']=='DM - CLD']['time_to_enroll'].dropna()), bins = 40, alpha = 0.3, density=1, label='CLD')
ax1.hist(np.array(progs[progs['PRGM_NM']=='DM - HF']['time_to_enroll'].dropna()), bins = 40, alpha = 0.3, density=1, label='HF')
ax1.set_ylabel('Percent of Patients')
ax1.set_xlabel('days to enrollment')
ax1.legend();


# In[ ]:


fighist_TOC = plt.figure(figsize=(12,6))
ax1 = fighist_TOC.add_subplot(111)
ax1.set_title('Histogram, Program Duration')
ax1.hist(np.array((progs[(progs['PRGM_NM']=='Transitions of Care - Post Discharge') &
                  (progs['is_optin']==0)]['prog_duration'].dropna())), bins = 100, alpha = 0.4, density=1, label='Opt-out')
ax1.hist(np.array((progs[(progs['PRGM_NM']=='Transitions of Care - Post Discharge') &
                  (progs['is_optin']==1)]['prog_duration'].dropna())), bins = 100, alpha = 0.4, density=1, label='Opt-in')
ax1.set_ylabel('Percent of Patients')
ax1.set_xlabel('days in program')
ax1.set_xlim(left=-1, right=50)
ax1.legend();


# In[ ]:


fighist_TOC = plt.figure(figsize=(12,6))
ax1 = fighist_TOC.add_subplot(111)
ax1.set_title('Histogram, Program Duration, TOC')
ax1.hist(np.array((progs[(progs['PRGM_NM']=='Transitions of Care - Post Discharge') &
                  (progs['is_optin']==0)]['time_to_enroll'].dropna())), bins = 150, alpha = 0.4, density=1, label='Opt-out')
ax1.hist(np.array((progs[(progs['PRGM_NM']=='Transitions of Care - Post Discharge') &
                  (progs['is_optin']==1)]['time_to_enroll'].dropna())), bins = 150, alpha = 0.4, density=1, label='Opt-in')
ax1.set_ylabel('Percent of Patients')
ax1.set_xlabel('days in program')
ax1.set_xlim(left=0, right=50)
ax1.legend();


# In[ ]:


fighist_TOC = plt.figure(figsize=(12,6))
ax1 = fighist_TOC.add_subplot(111)
ax1.set_title('Histogram, time to enroll (TOC)')
ax1.hist(np.array(progs[progs['PRGM_NM']=='Transitions of Care - Post Discharge']['time_to_enroll'].dropna()), bins = 200, alpha = 0.4, density=1, label='TOC')
ax1.set_ylabel('Percent of Patients')
ax1.set_xlabel('days to enrollment')
ax1.set_xlim(left=0, right=40)
ax1.legend();


# In[ ]:


progs.pivot_table(values='EMPI', index='PRGM_NM', aggfunc=['count'])


# In[ ]:


progs[progs['PRGM_NM']=='Transitions of Care - Post Discharge'].pivot_table(values='adm_30_after_TOC', index='PRGM_STOP_RSN', aggfunc=['count','mean'])

