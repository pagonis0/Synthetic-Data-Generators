import psycopg2
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
import statsmodels.api as sm
from statsmodels.formula.api import ols
from distfit import distfit
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from statsmodels.tsa import forecasting
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, plot_predict
from pmdarima.arima.utils import ndiffs
from statsmodels.tsa.stattools import acf
from sklearn.model_selection import train_test_split
plt.style.use('tableau-colorblind10')

df1 = pd.read_csv("/data/aimotion/OULAD/OULAD_processed.csv")
df1['finalgrade'] = df1['norm_finalgrade']*100
df1 = df1.drop('Unnamed: 0', axis=1)
df_trimmed1 = df1[df1['finalgrade'].between(0.001, 99.999)]

df2 = pd.read_csv("SLP_unit_processed_ogGrades.csv")
df2['finalgrade'] = (df2['full_score']-df2['score'])/df2['full_score']*100
df2 = df2.drop('Unnamed: 0', axis=1)
df_trimmed2 = df2[df2['finalgrade'].between(0.001, 99.999)]

def UserReindex(df, userid = 'userid'):
    if isinstance(df, pd.DataFrame):
        df_n = df.sort_values('userid', inplace=True)
        df_n = df.reset_index()
        user = df_n['userid'].to_list()
        from itertools import accumulate
        indexes  = range(len(user))
        byGroup  = accumulate(indexes,lambda i,u: (i+1)*(u>0 and user[u-1]==user[u]))
        indexes  = [i-1 for i in accumulate(int(g==0) for g in byGroup)]
        indexAndUser = [(i,u) for i,u in zip(indexes,user)]
        new_user = pd.DataFrame([(i,u) for i,u in zip(indexes,user)], columns=['new_user', 'old_user'])

        df_n['userid_n'] = new_user['new_user']
        df_n = df_n.drop('index', axis=1)
        return df_n
    else:
        raise TypeError('The imported object is not a pandas.DataFrame. Please import a pandas.DataFrame type.')
    

def bootstrap(data, samples=500, sample_size=500, stat='nostat'):
    my_samples = []
    for _ in range(samples):
        x = np.random.choice(data, size=sample_size, replace=True)
        if stat=='median':
            my_samples.append(np.median(x))
        elif stat=='nostat':
            my_samples.append(np.random.choice(x))
        else:
            my_samples.append(x.mean())
    return pd.Series(my_samples)


from tqdm import tqdm

# Student Simulator version 2.1
import random
def StudentSimulator(df, size, method, userid = 'userid', seed = 123, add_method = 'reuse'):
    
    np.random.seed(seed)
    vector = np.vectorize(np.int_)
    
    #Create the df
    data = pd.DataFrame(columns=["id", 'userid', 'itemid', 'courseid', 'finalgrade'])

    #Adding idx
    data['id'] = list(range(1,size+1))
    
    #Adding userid
    if add_method == 'reuse':
        #Creating the output dataframe
        newdf = pd.DataFrame()

        #Gather the unique users
        users = df[userid].unique()

        #Simulate students and reindexing
        ids = np.array(range(1, size+1))
        newuserid = []
        counter = 0

        while len(newdf) <= size:
            counter += 1
            user = int(np.random.choice(users, 1))
            t = df[userid]==user
            newdf = newdf.append(df[t]) 
            newuserid.append([counter]*t.sum())

        newuserid = [item for sublist in newuserid for item in sublist]
        newdf[userid] = newuserid

        newdf = newdf.reset_index()

        #Random remove the excess amount of observations 
        if len(newdf) > size:
            remove_n = len(newdf) - size
            drop_indices = np.random.choice(newdf.index, remove_n, replace=False)
            newdf1 = newdf.drop(drop_indices)
            
        users = newdf1['userid'].to_list()
        courses = newdf1['courseid'].to_list()
        items = newdf1['itemid'].to_list()
        data['userid'] = users
        data['courseid'] = courses
        data['itemid'] = items
        
        #check_nan = data['userid'].isnull().values.any()
        #print(check_nan)
        #check_nan = data['courseid'].isnull().values.any()
        #print(check_nan)
        #check_nan = data['quizid'].isnull().values.any()
        #print(check_nan)
     
    #Adding time
    time=stats.lognorm.rvs(s=1.22, loc=-10.20, scale=389.70, size=size)
    data['time_used'] = time
    
    #Adding grades

    all_dataframes1 = []
    all_dataframes1_grades = []
    
    data = data.sample(frac=1).reset_index(drop=True)
    
    user_df = data[['userid', 'itemid', 'courseid', 'finalgrade']]

    user_df = user_df.groupby('userid') 
    rg1 = user_df.size().max()
    
    for i in tqdm(range(rg1), desc='Finding grades'):
        current_df = pd.DataFrame(columns=['userid', 'itemid', 'courseid', 'finalgrade'])    
        for x in user_df.groups:
            if len(user_df.get_group(x))>i:
                n = pd.DataFrame(user_df.get_group(x))
                a = n.iloc[[i]]
                current_df = current_df.append(a, ignore_index=True)
        all_dataframes1.append(current_df)
        all_dataframes1_grades.append(current_df['finalgrade'])
    a = []
    for i in tqdm(range(len(all_dataframes_grades)), desc='Saving grades'):
        if i >= 1:
            for j in range(len(all_dataframes1_grades)):
                #all_dataframes1_grades[j] = all_dataframes1_grades[j].to_list()
                #print(all_dataframes1_grades)
                if method == 'random':
                    if i <= j:
                        for k in range(len(all_dataframes1_grades[j])):
                            all_dataframes1_grades[j][k] = np.random.choice(bootstrap(all_dataframes_grades[i], sample_size=10, samples=10))
                    else:
                        for k in range(len(all_dataframes1_grades[j])):
                            a = np.array(range(i))
                            all_dataframes1_grades[j][k] = np.random.choice(bootstrap(all_dataframes_grades[np.random.choice(a)], sample_size=10, samples=10)) 
                elif method == 'half_random':
                    if i <= j:
                        for k in range(len(all_dataframes1_grades[j])):
                            all_dataframes1_grades[j][k] = bootstrap(all_dataframes_grades[i], sample_size=10, samples=10).median()
                    else:
                        for k in range(len(all_dataframes1_grades[j])):
                            a = np.array(range(i))
                            all_dataframes1_grades[j][k] = bootstrap(all_dataframes_grades[np.random.choice(a)], sample_size=10, samples=10).median()
    for i in range(len(all_dataframes1)):
        all_dataframes1[i]['finalgrade'] = all_dataframes1[i]['finalgrade'].append(all_dataframes1_grades[i], ignore_index=True)
    data = pd.concat(all_dataframes1)
    data['finalgrade'] = data['finalgrade'] + np.random.normal(0, 3, len(data['finalgrade']))
    data = data.sample(frac=1).reset_index(drop=True)
    
    data['finalgrade'].where(data['finalgrade']<100, 100, inplace=True)
    data['finalgrade'].where(data['finalgrade']>=0, 0, inplace=True)
    
    return data


user_df = df_trimmed1[['userid', 'itemid', 'courseid', 'finalgrade']]

all_dataframes = []
all_dataframes_grades = []
user_df = df_trimmed1[['userid', 'itemid', 'courseid', 'finalgrade']]

user_df = user_df.groupby('userid') 
rg = user_df.size().max()-1

for i in tqdm(range(rg)):
    current_df = pd.DataFrame(columns=['userid', 'itemid', 'courseid', 'finalgrade'])    
    for x in user_df.groups:
        if len(user_df.get_group(x))>i:
            n = pd.DataFrame(user_df.get_group(x))
            a = n.iloc[[i]]
            current_df = current_df.append(a, ignore_index=True)
    all_dataframes.append(current_df)
    all_dataframes_grades.append(current_df['finalgrade'])

df_sim1 = StudentSimulator(df_trimmed1, len(df_trimmed1), method = 'half_random')
df_sim1.to_csv('simdata1.csv')

df_sim1_2 = StudentSimulator(df_trimmed1, len(df_trimmed1), method = 'random')
df_sim1_2.to_csv('simdata1_rand.csv')



user_df = df_trimmed2[['userid', 'itemid', 'courseid', 'finalgrade']]  

all_dataframes = []
all_dataframes_grades = []
user_df = df_trimmed2[['userid', 'itemid', 'courseid', 'finalgrade', 'timemodified']]

user_df = user_df.sort_values('timemodified', ascending=True).groupby('userid') 
rg = user_df.size().max()-1

for i in tqdm(range(rg)):
    current_df = pd.DataFrame(columns=['userid', 'itemid', 'courseid', 'finalgrade'])    
    for x in user_df.groups:
        if len(user_df.get_group(x))>i:
            n = pd.DataFrame(user_df.get_group(x))
            a = n.iloc[[i]]
            current_df = current_df.append(a, ignore_index=True)
    all_dataframes.append(current_df)
    all_dataframes_grades.append(current_df['finalgrade'])

df_sim2 = StudentSimulator(df_trimmed2, len(df_trimmed2), method = 'half_random')
df_sim2.to_csv('simdata2.csv')

df_sim2_2 = StudentSimulator(df_trimmed2, len(df_trimmed2), method = 'random')
df_sim2_2.to_csv('simdata2_rand.csv')