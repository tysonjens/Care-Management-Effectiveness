
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
# import warnings; warnings.simplefilter('ignore')
pd.set_option("display.max_columns", 100)
pd.set_option('display.max_rows', 500)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[263]:


## READ In Programs, Admissions, and Lace data
## Could have script run this
progs = pd.read_csv('data/Program_Patdim.csv')


# In[264]:


admits = pd.read_csv('data/admissions.csv', sep='|', low_memory=False)


# In[265]:


LACE = pd.read_csv('data/lace_empi', sep='|')


# In[266]:


assess = pd.read_csv('data/assessments.csv.sql', sep='|', low_memory=False)


# In[267]:


assess_toc_mcg_dmg =pd.read_csv('data/TOC_MCG_DMG_ASES_ONLY.rpt', sep='|', low_memory=False)


# In[268]:


## Cleaning, field removal, and data type changes


# In[269]:


## convert dates to date time
progs['prog_create_date'] = pd.to_datetime(progs['CRT_TMS'])
progs['date_of_birth'] = pd.to_datetime(progs['DOB'])
progs['prog_end_date'] = pd.to_datetime(progs['END_TMS'])


# In[270]:


## replace sex with is_male
opt_in = {'Admit to Facility':1, 'Discharge to Custodial':1, 'Discharged to Hospice':1, 
          'Discharged to PCP':1, 'Goals Met':1, 'Referred to other program':1, 
          'Refused':0, 'Unable to Reach':0, 'Barriers to participation':0, 'Expired':np.nan, 
         'Criteria not met':np.nan, 'Disenrolled from HP/Medical Group':np.nan, '6 months/1 year post transplant':np.nan,
       'Pharmacy - physician recommended':np.nan}
progs['is_optin'] = progs['PRGM_STOP_RSN'].replace(opt_in)
# progs['is_male'].fillna(0, inplace=True)


# In[271]:


admits['EMPI'].fillna(999999999, inplace=True)

admits['EMPI'] = admits['EMPI'].astype(int)


# In[273]:


## replace date_of _ birth with age
progs['age'] = datetime.now() - progs['date_of_birth']
progs['age'] = progs['age'] / timedelta(days=1) / 365
progs['age'].fillna(float(progs['age'].mean()), inplace=True)


# In[274]:


new_sex = {'F':0, 'M':1}
progs['is_male'] = progs['Sex'].replace(new_sex)
progs['is_male'].fillna(0, inplace=True)


# In[275]:


## drop unneeded columns
prog_cols_drop = ['PTNT_DK', 'PTNT_DK.1', 'DOB', 'Sex', 'CRT_TMS', 'END_TMS', 'TNT_MKT_BK', 'date_of_birth', 'PRGM_STS']
progs = progs.drop(prog_cols_drop, axis=1)


# In[276]:


progs = progs[(progs['prog_create_date']>'2018-04-01') & (progs['prog_create_date']<'2018-08-01')]


# In[277]:


progs = progs.reset_index()


# In[278]:


## convert object columns to categoricals
for col in ['RGON_NM', 'HP_NM', 'LOB_SUB_CGY', 'PRGM_NM', 'PRGM_STOP_RSN']:
    progs[col] = progs[col].astype('category')


# In[279]:


## convert dates to date time
admits['admit_date'] = pd.to_datetime(admits['AdmitDt'])
admits['discharge_date'] = pd.to_datetime(admits['DischDt'])


# In[280]:


## replace Model with is_group
new_Model = {'GROUP':1, 'IPA':0, 'JOINT VENTURE':0, 'OPEN ACCESS':0, 'OTHER':0}
admits['is_group'] = admits['Model'].replace(new_Model)
admits['is_group'].fillna(0, inplace=True)


# In[281]:


## drop unneeded columns
admits_cols_drop = ['Model', 'PCP', 'MM', 'MRN', 'LastName',
       'FirstName', 'AdmitDt', 'DischDt']
admits = admits.drop(admits_cols_drop, axis=1)


# In[282]:


assess['ASES_DT'] = pd.to_datetime(assess['ASES_DT'])
assess['EFF_FM_TS'] = pd.to_datetime(assess['EFF_FM_TS'])


# In[283]:


assess_toc_mcg_dmg['ASES_DT'] = pd.to_datetime(assess_toc_mcg_dmg['ASES_DT'])
assess_toc_mcg_dmg['EFF_FM_TS'] = pd.to_datetime(assess_toc_mcg_dmg['EFF_FM_TS'])


# In[284]:


new_text = {'Successful (enter number you\'re calling)':'Success'}
assess['ANSR_TXT'] = assess['ANSR_TXT'].replace(new_text)


# In[285]:


## change objects to categories
cols_to_category = ['REGION', 'SITE', 'LOB', 'Acuity', 'Facility',
       'RefType', 'DayType', 'AdmissionType', 'DischDx1', 'DischDx1Desc', 'SurgeryPx', 'DISPOSITION', 'ACSA_CAT']
for col in cols_to_category:
    admits[col] = admits[col].astype('category')


# In[286]:


## update LACE score data
LACE['ASES_DT'] = pd.to_datetime(LACE['ASES_DT'])


# In[287]:


## create new columns using functions and the admits data


# In[288]:


admits_all = admits


# In[289]:


## only keep acute admissions
admits = admits[admits['Acuity']=='ACUTE']


# In[290]:


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


# In[291]:


index_dates = find_index_admit(progs, admits_all)
progs['index_date'] = index_dates
progs['index_date'] = pd.to_datetime(progs['index_date'])


# In[292]:


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


# In[293]:


first_admissions = find_first_admission_after_enroll(progs, admits)
progs['frst_adm_aftr_enrl'] = first_admissions
progs['frst_adm_aftr_enrl'] = pd.to_datetime(progs['frst_adm_aftr_enrl'])


# In[294]:


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


# In[295]:


lace_scores = find_lace_prior_to_enroll(progs, LACE)
progs['lace_score'] = lace_scores


# In[296]:


progs[(progs['PRGM_NM']=='Transitions of Care - Post Discharge') | 
      (progs['PRGM_NM']=='Transitions of Care - ED')].count()


# In[297]:


len(set(assess['PTNT_ASES_DK']))


# In[298]:


progs['lace_score'].fillna(6, inplace=True)


# In[299]:


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


# In[300]:


def adms_to_one_zero_v2(ases):
    ases_yn = []
    for num in ases:
        if num>0:
            ases_yn.append(1)
        else:
            ases_yn.append(0)
    return ases_yn


# In[301]:


cm_toc_ases = find_assessments_during_program(progs, assess_toc_mcg_dmg, ases_nm='MCG - DMG - Post-Hospitalization Follow-Up')


# In[302]:


len(cm_toc_ases)


# In[303]:


cm_toc_ases = adms_to_one_zero_v2(cm_toc_ases)


# In[304]:


len(cm_toc_ases)


# In[305]:


progs['cm_touch_yn'] = cm_toc_ases


# In[306]:


ases_nm = 'CM Medication Reconciliation'


# In[310]:


## subset assessment to only include questions where the answer = "Success"
success_ases = assess[assess['ANSR_TXT']=='Success']
TOCPDO_cnt = find_assessments_during_program(progs, success_ases)
progs['success_TOCPDO_cnt'] = TOCPDO_cnt


# We hope to include "Medication Reconciliation" and "Post Hospitalization Follow-up" to the model, but currently these assessments only date back to mid-August, and therefore don't cover patients in TOC since April.

# In[311]:


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


# In[312]:


assess.columns


# In[313]:


list(set(assess['EMPI'][0:20]))


# In[314]:


docrev_dissum = feature_from_assessment_text(progs, assess, ases_nm='TOC Post Discharge Outreach',
                                 qstn_txt='Indicate the Documents Reviewed',
                                ansr_txt='Discharge Summary')
progs['docrev_dissum'] = docrev_dissum


# In[ ]:


docrev_histphys = feature_from_assessment_text(progs, assess, ases_nm='TOC Post Discharge Outreach',
                                 qstn_txt='Indicate the Documents Reviewed',
                                ansr_txt='History and Physical')
progs['docrev_histphys'] = docrev_histphys


# In[ ]:


docrev_medrec_ehr = feature_from_assessment_text(progs, assess, ases_nm='TOC Post Discharge Outreach',
                                 qstn_txt='Indicate the Documents Reviewed',
                                ansr_txt='Medication reconciliation completed in TWEHR')
progs['docrev_medrec_ehr'] = docrev_medrec_ehr


# In[ ]:


docrev_meds = feature_from_assessment_text(progs, assess, ases_nm='TOC Post Discharge Outreach',
                                 qstn_txt='Indicate the Documents Reviewed',
                                ansr_txt='Medications')
progs['docrev_meds'] = docrev_meds


# In[ ]:


docrev_PCPrecs = feature_from_assessment_text(progs, assess, ases_nm='TOC Post Discharge Outreach',
                                 qstn_txt='Indicate the Documents Reviewed',
                                ansr_txt='PCP Records')
progs['docrev_PCPrecs'] = docrev_PCPrecs


# In[ ]:


docrev_problist = feature_from_assessment_text(progs, assess, ases_nm='TOC Post Discharge Outreach',
                                 qstn_txt='Indicate the Documents Reviewed',
                                ansr_txt='Problems List')
progs['docrev_problist'] = docrev_problist


# In[ ]:


docrev_refhist = feature_from_assessment_text(progs, assess, ases_nm='TOC Post Discharge Outreach',
                                 qstn_txt='Indicate the Documents Reviewed',
                                ansr_txt='Referral History')
progs['docrev_refhist'] = docrev_refhist


# In[ ]:


no_part_CM = feature_from_assessment_text(progs, assess, ases_nm='TOC Post Discharge Outreach',
                                 qstn_txt='Patient is willing to participate in Care Management?',
                                ansr_txt='No')
progs['no_part_CM'] = no_part_CM


# In[ ]:


no_part_DM = feature_from_assessment_text(progs, assess, ases_nm='TOC Post Discharge Outreach',
                                 qstn_txt='Patient is willing to participate in Care or Disease Management and agrees to the recommended contact interval. The program was explained to the patient and they understand it is optional and that they can opt out at any time.',
                                ansr_txt='No')
progs['no_part_DM'] = no_part_DM


# In[ ]:


agree_care_plan = feature_from_assessment_text(progs, assess, ases_nm='TOC Post Discharge Outreach',
                                 qstn_txt='Patient verbalized agreement to Care Plan',
                                ansr_txt='Yes')
progs['agree_care_plan'] = agree_care_plan


# In[ ]:


assess_success = assess[(assess['ANSR_TXT']=='Continue TOC') | (assess['ANSR_TXT']=='Success')]


# In[ ]:



##medrec_cnt = find_assessments_during_program(progs,assess, ases_nm='CM Medication Reconciliation')
##progs['medrec_cnt']=medrec_cnt


# In[315]:


##phfu_cnt = find_assessments_during_program(progs ,assess, ases_nm='Post-Hospitalization Follow-up')
##progs['phfu_cnt']=phfu_cnt


# In[316]:


## function that counts the number of admits in a window AFTER a program begins
def get_adm_after(programs, admissions, window_size=90):
    admits_in_window = list(np.zeros(programs.shape[0]))
    for index, row in programs.iterrows():
        admit_pat = admissions[admissions['EMPI']==row['EMPI']]
        count = 0
        for index2, row2 in admit_pat.iterrows():
            if (row2['admit_date'] < (row['prog_create_date']+timedelta(days=window_size))) & (row2['admit_date'] > row['prog_create_date']):
                count+=1
        admits_in_window[index] = count
    return admits_in_window


# In[317]:


## calc number of admits that occur within 30 day window after program begins
thirty_day_after = get_adm_after(progs, admits)
progs['adm_90_after'] = thirty_day_after


# In[320]:


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


# In[321]:


thirty_day_after_TOC = get_adm_after_TOC(progs, admits)
progs['adm_30_after_TOC'] = thirty_day_after_TOC


# In[ ]:


## function that counts relevant touches during a care management program
#def count_touches_that_count(programs, assessments):
#    touches_that_count = list(np.zeros(programs.shape[0]))
#    for index, row in programs.iterrows():
#        assess


# In[322]:


## function that counts the number of admits in a window BEFORE a program begins
def get_adm_before(programs, admissions, window_size=90):
    admits_in_window = list(np.zeros(programs.shape[0]))
    for index, row in programs.iterrows():
        admit_pat = admissions[admissions['EMPI']==row['EMPI']]
        count = 0
        for index2, row2 in admit_pat.iterrows():
            if (row2['admit_date'] > (row['prog_create_date']-timedelta(days=window_size))) & (row2['admit_date'] < row['prog_create_date']):
                count+=1
        admits_in_window[index] = count
    return admits_in_window


# In[323]:


## calc number of admits that occur within 30 day window before program begins
thirty_day_before = get_adm_before(progs, admits)
progs['adm_90_before'] = thirty_day_before


# In[324]:


progs['time_to_enroll'] = progs['prog_create_date']-progs['index_date']
progs['time_to_enroll'] = progs['time_to_enroll']/ timedelta(days=1)


# In[325]:


progs['time_to_enroll'].fillna(float(progs['time_to_enroll'].mean()), inplace=True)


# In[326]:


progs['prog_duration'] = progs['prog_end_date']-progs['prog_create_date']
progs['prog_duration'] = progs['prog_duration']/ timedelta(days=1)


# In[327]:


progs['prog_duration'].fillna(float(progs['prog_duration'].mean()), inplace=True)


# In[328]:


progs['index_to_next_days'] = progs['frst_adm_aftr_enrl']-progs['index_date']
progs['index_to_next_days'] = progs['index_to_next_days']/ timedelta(days=1)


# In[329]:


def adms_to_one_zero(progs):
    adm_yn = []
    for index, row in progs.iterrows():
        if row['adm_30_after_TOC']>0:
            adm_yn.append(1)
        else:
            adm_yn.append(0)
    return adm_yn


# In[330]:


adm_yn = adms_to_one_zero(progs)


# In[331]:


progs['is_30_TOC_adm'] = adm_yn


# In[ ]:


fighist = plt.figure(figsize=(12,6))
ax1 = fighist.add_subplot(111)
ax1.set_title('Histogram, Days Program Duration')
#ax1.hist(np.array(progs[progs['PRGM_NM']=='Transitions of Care - Post Discharge']['time_to_enroll'].dropna()), bins = 100, alpha = 0.4, density=1, label='TOC')
ax1.hist(np.array(progs[progs['PRGM_NM']=='DM - CLD']['prog_duration'].dropna()), bins = 150, alpha = 0.6, density=1, label='CLD')
ax1.hist(np.array(progs[progs['PRGM_NM']=='DM - HF']['prog_duration'].dropna()), bins = 150, alpha = 0.6, density=1, label='HF')
ax1.set_ylabel('Percent of Patients')
ax1.set_xlabel('Days in Program')
ax1.set_xlim(left=-5, right=150)
ax1.legend();


# In[ ]:


fighist = plt.figure(figsize=(12,6))
ax1 = fighist.add_subplot(111)
ax1.set_title('Histogram, time to enroll post discharge')
#ax1.hist(np.array(progs[progs['PRGM_NM']=='Transitions of Care - Post Discharge']['time_to_enroll'].dropna()), bins = 100, alpha = 0.4, density=1, label='TOC')
ax1.hist(np.array(progs[progs['PRGM_NM']=='DM - CLD']['time_to_enroll'].dropna()), bins = 40, alpha = 0.6, density=1, label='CLD')
ax1.hist(np.array(progs[progs['PRGM_NM']=='DM - HF']['time_to_enroll'].dropna()), bins = 40, alpha = 0.6, density=1, label='HF')
ax1.set_ylabel('Percent of Patients')
ax1.set_xlabel('days to enrollment')
ax1.legend();


# In[ ]:


fighist_TOC = plt.figure(figsize=(12,6))
ax1 = fighist_TOC.add_subplot(111)
ax1.set_title('Histogram, Program Duration')
ax1.hist(np.array((progs[(progs['PRGM_NM']=='Transitions of Care - Post Discharge') &
                  (progs['is_optin']==0)]['prog_duration'].dropna())), bins = 100, alpha = 0.6, density=1, color='grey', label='Opt-out')
ax1.hist(np.array((progs[(progs['PRGM_NM']=='Transitions of Care - Post Discharge') &
                  (progs['is_optin']==1)]['prog_duration'].dropna())), bins = 100, alpha = 0.6, density=1, color='green', label='Opt-in')
ax1.set_ylabel('Percent of Patients')
ax1.set_xlabel('days in program')
ax1.set_xlim(left=-5, right=50)
ax1.legend();


# In[ ]:


fighist_TOC = plt.figure(figsize=(12,6))
ax1 = fighist_TOC.add_subplot(111)
ax1.set_title('Histogram, LACE Scores by Program Participation, TOC')
ax1.hist(np.array((progs[(progs['PRGM_NM']=='Transitions of Care - Post Discharge') &
                  (progs['is_optin']==0)]['lace_score'].dropna())), 
         bins = 18, alpha = 0.6, density=1, color='grey', label='Opt-out')
ax1.hist(np.array((progs[(progs['PRGM_NM']=='Transitions of Care - Post Discharge') &
                  (progs['is_optin']==1)]['lace_score'].dropna())), 
         bins = 18, alpha = 0.6, density=1, color='green', label='Opt-in')
ax1.set_ylabel('Percent of Patients')
ax1.set_xlabel('Lace Score')
ax1.set_xlim(left=0, right=20)
ax1.legend();


# In[ ]:


fighist_TOC = plt.figure(figsize=(12,6))
ax1 = fighist_TOC.add_subplot(111)
ax1.set_title('Histogram, Days to Enrollment, TOC')
ax1.hist(np.array((progs[(progs['PRGM_NM']=='Transitions of Care - Post Discharge') &
                  (progs['is_optin']==0)]['time_to_enroll'].dropna())), bins = 150, alpha = 0.6, density=1, color='grey', label='Opt-out')
ax1.hist(np.array((progs[(progs['PRGM_NM']=='Transitions of Care - Post Discharge') &
                  (progs['is_optin']==1)]['time_to_enroll'].dropna())), bins = 150, alpha = 0.6, density=1, color='green', label='Opt-in')
ax1.set_ylabel('Percent of Patients')
ax1.set_xlabel('Days to Enrollment')
ax1.set_xlim(left=0, right=50)
ax1.legend();


# In[ ]:


fighist_TOC = plt.figure(figsize=(12,6))
ax1 = fighist_TOC.add_subplot(111)
ax1.set_title('Histogram, time to enroll (TOC)')
ax1.hist(np.array(progs[progs['PRGM_NM']=='Transitions of Care - Post Discharge']['time_to_enroll'].dropna()), bins = 200, alpha = .6, density=1, color='green', label='TOC')
ax1.set_ylabel('Percent of Patients')
ax1.set_xlabel('days to enrollment')
ax1.set_xlim(left=0, right=40)
ax1.legend();


# In[ ]:


fighist_TOC = plt.figure(figsize=(12,6))
ax1 = fighist_TOC.add_subplot(111)
ax1.set_title('Scatterplot, TOC Ages by Lace Scores')
ax1.scatter(np.array(progs[progs['PRGM_NM']=='Transitions of Care - Post Discharge']['age'].dropna()),
            np.array(progs[progs['PRGM_NM']=='Transitions of Care - Post Discharge']['lace_score'].dropna()))
ax1.set_ylabel('Lace Score')
ax1.set_xlabel('Age')
#ax1.set_xlim(left=0, right=40)
#ax1.legend()
("")


# In[332]:


progs[progs['PRGM_NM']=='Transitions of Care - Post Discharge'].pivot_table(values='adm_30_after_TOC', index='PRGM_STOP_RSN', aggfunc=['count','mean'], dropna=True)


# In[ ]:


## Understand relationship between LACE and time to enroll


# In[333]:


progs[progs['PRGM_NM']=='Transitions of Care - Post Discharge'].pivot_table(values='adm_30_after_TOC', index='lace_score', columns='is_optin', aggfunc=['count','mean'], dropna=True)


# In[ ]:


## graph violinplots of optin vs. optout LACE Scores.
fighist_lace_age = plt.figure(figsize=(12,6))
ax1 = fighist_lace_age.add_subplot(111)
sns.boxplot(ax=ax1, x="lace_score", y='age', data=progs[progs['PRGM_NM']=='Transitions of Care - Post Discharge'])


# In[ ]:


## graph violinplots of optin vs. optout LACE Scores.
sns.violinplot(x="is_optin", y='lace_score', data=progs[progs['PRGM_NM']=='Transitions of Care - Post Discharge'])


# In[ ]:


## CHF difference of differences analysis.  


# In[ ]:


## use logisic regression to test whether opt_in has impact on readmit_30_days_TOC, controlling for other variables.


# In[ ]:


## need to fill X vars' na with appropriate values:
## is_optin in - drop na rows
## age - average - check
## is_male - make them 0s - check
## lace_score = average lace score - check
## time_to_enroll = average - check
## prog duration - average


# In[353]:


progs = progs[progs['is_optin'].isna()==False]


# In[ ]:


progs.head()


# In[335]:


model = linear_model.LogisticRegression(C=100000)
y = np.array(progs['is_30_TOC_adm'])
X = np.array(progs[['cm_touch_yn', 'age', 'is_male', 'lace_score', 'time_to_enroll', 'prog_duration']])


# In[336]:


model.fit(X, y)


# In[337]:


model.coef_


# In[338]:


coefs = model.coef_


# In[339]:


coefs


# In[340]:


np.exp(coefs)


# In[341]:


inter = model.intercept_


# In[342]:


inter


# In[351]:


np.exp((coefs[0][0:5] * [0,70,0,8,2]).sum() + inter)


# In[349]:


.12/1.12


# In[354]:


progs.pivot_table('index', 'is_optin', 'cm_touch_yn', aggfunc='count')


# In[ ]:


np.eaxp(coefs)


# In[ ]:


## two to 1 odds means twice as likely than not - 66.6%  -- google how to interpret log odds and double check that
## logisticRegress() returns log odds.


# In[ ]:


## Hypothesis: Of discharged patients, most readmissions happen in the first 10 days. If this is true, how does it impact
## our ability to use time cutoffs to groups patients into opt-ins or opt-outs?


# In[ ]:


fighist_TOC = plt.figure(figsize=(12,6))
ax1 = fighist_TOC.add_subplot(111)
ax1.set_title('Histogram, Discharge to Next Admission, days (TOC)')
ax1.hist(np.array(progs[progs['PRGM_NM']=='Transitions of Care - Post Discharge']['index_to_next_days'].dropna()), bins = 150, alpha = 0.8, density=1, label='TOC')
ax1.set_ylabel('Percent of Patients')
ax1.set_xlabel('days to next admission')
ax1.set_xlim(left=0, right=200)
ax1.legend();


# In[ ]:


## Initial numbers for HF Difference of Differences


# In[ ]:


progs[(progs['PRGM_NM']=='DM - HF') & 
      (progs['is_optin']==1) &
      (progs['prog_create_date']<'2018-05-01')]['adm_90_before'].mean()


# In[ ]:


progs[(progs['PRGM_NM']=='DM - HF') & 
      (progs['is_optin']==1) &
      (progs['prog_create_date']<'2018-05-01')]['adm_90_after'].mean()


# In[ ]:


progs[(progs['PRGM_NM']=='DM - HF') & 
      (progs['is_optin']==0) &
      (progs['prog_create_date']<'2018-05-01')]['adm_90_before'].mean()


# In[ ]:


progs[(progs['PRGM_NM']=='DM - HF') & 
      (progs['is_optin']==0) &
      (progs['prog_create_date']<'2018-05-01')]['adm_90_after'].mean()


# In[ ]:


progs[(progs['PRGM_NM']=='DM - CLD') & 
      (progs['is_optin']==1) &
      (progs['prog_create_date']<'2018-05-01')]['adm_90_before'].mean()


# In[ ]:


progs[(progs['PRGM_NM']=='DM - CLD') & 
      (progs['is_optin']==1) &
      (progs['prog_create_date']<'2018-05-01')]['adm_90_after'].mean()


# In[ ]:


progs[(progs['PRGM_NM']=='DM - CLD') & 
      (progs['is_optin']==0) &
      (progs['prog_create_date']<'2018-05-01')]['adm_90_before'].mean()


# In[ ]:


progs[(progs['PRGM_NM']=='DM - CLD') & 
      (progs['is_optin']==0) &
      (progs['prog_create_date']<'2018-05-01')]['adm_90_after'].mean()


# In[ ]:


progs[(progs['PRGM_NM']=='DM - CLD') & 
      (progs['prog_create_date']<'2018-05-01')].pivot_table(values='adm_90_after', index='is_optin', aggfunc='mean') 


# In[ ]:


progs.to_csv('data/progs_clean.csv')

