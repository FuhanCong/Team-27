#!/usr/bin/env python
# coding: utf-8

# ### https://github.com/FuhanCong/Team-27

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[54]:


# A function that presents all initial EDA of dataset
def initial_eda(df):
    if isinstance(df, pd.DataFrame):
        total_na = df.isna().sum().sum()
        print("Dimensions : %d rows, %d columns" % (df.shape[0], df.shape[1]))
        print("Total NA Values : %d " % (total_na))
        print("%38s %10s     %10s %10s" % ("Column Name", "Data Type", "#Distinct", "NA Values"))
        col_name = df.columns
        dtyp = df.dtypes
        uniq = df.nunique()
        na_val = df.isna().sum()
        for i in range(len(df.columns)):
            print("%38s %10s   %10s %10s" % (col_name[i], dtyp[i], uniq[i], na_val[i]))
        
    else:
        print("Expect a DataFrame but got a %15s" % (type(df)))


# In[55]:


initial_eda(protein_train)


# In[56]:


initial_eda(peptides_train)


# In[ ]:





# # Clinical_train (Disease severity)
# updrs_test - The patient's score for part N of the Unified Parkinson's Disease Rating Scale
# 
# UPDRS_1: non-motor experiences of daily living. Range 0-52. 10 and below is mild, 22 and above is severe
# 
# UPDRS_2: motor experiences of daily living. Range: 0–52. 12 and below is mild, 30 and above is severe
# 
# UPDRS_3: motor examination. Range: 0–132. 32 and below is mild, 59 and above is severe
# 
# UPDRS_4: motor complications related to motor function. Range: 0–24, 4 and below is mild, 13 and above is severe
# 
# upd23b_clinical_state_on_medication - Whether or not the patient was taking medication such as Levodopa during the UPDRS assessment.

# In[162]:


initial_eda(clinical_train) # collected by visit times


# In[163]:


clinical_train = pd.read_csv('/Users/jessie817/Desktop/ORIE5741/group project/amp-parkinsons-disease-progression-prediction (1)/train_clinical_data.csv')

protein_train = pd.read_csv('/Users/jessie817/Desktop/ORIE5741/group project/amp-parkinsons-disease-progression-prediction (1)/train_proteins.csv')
protein_test_train = pd.read_csv('/Users/jessie817/Desktop/ORIE5741/group project/amp-parkinsons-disease-progression-prediction (1)/test_proteins.csv')

peptides_train = pd.read_csv('/Users/jessie817/Desktop/ORIE5741/group project/amp-parkinsons-disease-progression-prediction (1)/train_peptides.csv')
peptides_test = pd.read_csv('/Users/jessie817/Desktop/ORIE5741/group project/amp-parkinsons-disease-progression-prediction (1)/test_peptides.csv')


# In[164]:


for i in ['updrs_1', 'updrs_2', 'updrs_3', 'updrs_4']:
    # x.mode().mean() - stable?
    mode = clinical_train.groupby('patient_id')[i].apply(lambda x : x.mode().mean())
    for j in mode.index:
        clinical_train.loc[clinical_train['patient_id'] == j, i] = clinical_train.loc[clinical_train['patient_id'] == j, i].fillna(mode[j])
initial_eda(clinical_train)
clinical_train


# In[165]:


initial_eda(clinical_train)


# In[235]:


clinical_train.loc[clinical_train['updrs_1'].between(0, 10, 'both'), 'updrs_1_rating'] = 'Mild'
clinical_train.loc[clinical_train['updrs_1'].between(10, 21, 'right'), 'updrs_1_rating'] = 'Moderate'
clinical_train.loc[clinical_train['updrs_1'].between(21, 52, 'right'), 'updrs_1_rating'] = 'Severe'

clinical_train.loc[clinical_train['updrs_2'].between(0, 12, 'both'), 'updrs_2_rating'] = 'Mild'
clinical_train.loc[clinical_train['updrs_2'].between(12,29, 'right'), 'updrs_2_rating'] = 'Moderate'
clinical_train.loc[clinical_train['updrs_2'].between(29, 52, 'right'), 'updrs_2_rating'] = 'Severe'

clinical_train.loc[clinical_train['updrs_3'].between(0, 32, 'both'), 'updrs_3_rating'] = 'Mild'
clinical_train.loc[clinical_train['updrs_3'].between(32,58, 'right'), 'updrs_3_rating'] = 'Moderate'
clinical_train.loc[clinical_train['updrs_3'].between(58, 132, 'right'), 'updrs_3_rating'] = 'Severe'

clinical_train.loc[clinical_train['updrs_4'].between(0, 4, 'both'), 'updrs_4_rating'] = 'Mild'
clinical_train.loc[clinical_train['updrs_4'].between(4,12, 'right'), 'updrs_4_rating'] = 'Moderate'
clinical_train.loc[clinical_train['updrs_4'].between(12, 24, 'right'), 'updrs_4_rating'] = 'Severe'


# In[233]:


f, axs = plt.subplots(2, 2, figsize = (8, 6))
for i, ax in zip(['updrs_1', 'updrs_2', 'updrs_3', 'updrs_4'], axs.ravel()):
    data = pd.DataFrame(clinical_train[i].value_counts()).reset_index()
    sns.histplot(data = clinical_train[i], kde = True, ax = ax)
    ax.set_title(i)
    ax.set_xlabel('score')
    ax.set_ylabel('count')
plt.subplots_adjust(wspace = 0.5, hspace = 0.5)
plt.rcParams["xtick.labelsize"] = 9
plt.show()
# right skewed: maybe more serious, fewer people(die)


# In[188]:


clinical_train.columns


# In[236]:


# transfer clinical_train from wide table to long table
clinical_train_long = pd.melt(clinical_train[['visit_id', 'patient_id', 'visit_month','updrs_1_rating', 'updrs_2_rating', 'updrs_3_rating', 'updrs_4_rating']]
                              , id_vars=['visit_id', 'patient_id', 'visit_month']
                              , var_name='updrs_type', value_name='updrs_rating')
clinical_train_long = clinical_train_long.groupby(['updrs_type', 'updrs_rating']).count().reset_index()
clinical_train_long


# In[260]:


plt.figure(figsize = (10, 6))
ax = sns.barplot(data = clinical_train_long, x = 'updrs_type', y = 'patient_id', hue = 'updrs_rating', palette = "Blues")
plt.title('Number of patients of different UPDRS ratings')
plt.legend(loc = 'upper right')
ax.bar_label(ax.containers[0])
ax.bar_label(ax.containers[1])
ax.bar_label(ax.containers[2])
plt.show()
# also right skewed


# In[283]:


month_uprs.columns


# In[297]:


month_uprs = clinical_train.groupby('visit_month').mean().reset_index()
month_uprs.head()


# In[296]:


plt.figure(figsize = (8, 6))
for i in ['updrs_1', 'updrs_2','updrs_3', 'updrs_4']:
    plt.plot('visit_month', i, data = month_uprs)
plt.title('Trend of UPDRS_mean by visiting month')
plt.legend(loc = 'upper left')
plt.xticks(list(month_uprs['visit_month']), list(month_uprs['visit_month'])) # months are not evenly distributed
plt.show()
# The scores are not a linear relationship with months.


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




