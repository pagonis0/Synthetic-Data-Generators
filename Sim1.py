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
    



# Dataset: /data/aimotion/OULAD/OULAD_processed.csv
# Trimmed Dataset
# Student Simulator 1

df1 = pd.read_csv("/data/aimotion/OULAD/OULAD_processed.csv")
df1['finalgrade'] = df1['norm_finalgrade']*100
df1 = df1.drop('Unnamed: 0', axis=1)
df_trimmed1 = df1[df1['finalgrade'].between(0.001, 99.999)]

def StudentSimulator(df, size, add_method="reuse", userid = 'userid', seed=123):
    import random
    #Create the df
    data = pd.DataFrame(columns=['userid', 'itemid', 'norm_finalgrade', 'time_diff', 'timemodified', 'courseid', 'finalgrade'])
    
    np.random.seed(seed)
    vector = np.vectorize(np.int_)
    
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
       
    elif add_method == "simulate":
        #Adding userid accoring to statistics (mean=7, std=10)
        df = UserReindex(df = df, userid = userid)
        vector = np.vectorize(np.int_)
        uniques = int(size*df['userid_n'].nunique()/len(df))
        sampler = vector(np.linspace(start=1, stop=uniques, num=uniques))

        users = df['userid_n'].value_counts().rename_axis('userid').reset_index(name='counts')
        users_count = users['counts'].value_counts().rename_axis('no_of_grades').reset_index(name='no_of_students') 
        users_count = users_count.sort_values('no_of_grades')

        prob = stats.pareto.rvs(size=uniques, b=1.86, loc=-6.788, scale=7.788, random_state=seed)
        prob /= np.sum(prob)

        sampled_users = pd.Series(np.random.choice(sampler, size, p=prob))    
        data['userid'] = sampled_users


        #Adding courseid
        uniques = int(size*df['course'].nunique()/len(df))
        sampler = vector(np.linspace(start=1, stop=uniques, num=uniques))

        courses = df['courseid'].value_counts().rename_axis('courseid').reset_index(name='counts')
        courses_count = courses['counts'].value_counts().rename_axis('no_of_grades').reset_index(name='no_of_courses') 
        courses_count = courses_count.sort_values('no_of_grades')

        prob = stats.expon.rvs(size=uniques, loc=1, scale=58.928, random_state=seed)
        prob /= np.sum(prob)

        sampled_courses = pd.Series(np.random.choice(sampler, size, p=prob))    
        data['courseid'] = sampled_courses    

        #Adding items

        items = list(range(1, size+1))
        sampled_items = []
        data = data.sort_values('courseid')
        for i in tqdm(data['courseid'].unique(), desc='Finding quizzes'):
            myitems = [items.pop(random.randrange(len(items))) for _ in range(int(max(1, 0.15*len(data[data['courseid']==i]))))]
            sampled_items.append([random.choice(myitems) for _ in range(len(data[data['courseid']==i]))])


        data['itemid'] = sum(sampled_items, [])  

    

    #Adding grades
    grade=stats.loggamma.rvs(c=0.502433, loc=89.017, scale=8.19915, size=size)
    data['finalgrade'] = grade
    data['finalgrade'].where(data['finalgrade'] >= 0, 0, inplace=True)
    data['finalgrade'].where(data['finalgrade'] < 100, 100, inplace=True)

    data['norm_finalgrade'] = data['finalgrade']/100

    #Adding time
    time=stats.uniform.rvs(loc=12, scale=249, size=size)
    data['time_diff'] = time

    return(data)

simdata = StudentSimulator(df_trimmed1, len(df_trimmed1))
simdata.to_csv('/data/aimotion/OULAD/StuSi_OULAD_trimmed_full.csv')
simdata = StudentSimulator(df_trimmed1, 2*len(df_trimmed1))
simdata.to_csv('/data/aimotion/OULAD/StuSi_OULAD_trimmed_double.csv')
simdata = StudentSimulator(df_trimmed1, 3*len(df_trimmed1))
simdata.to_csv('/data/aimotion/OULAD/StuSi_OULAD_trimmed_triple.csv')


# Dataset: /data/aimotion/OULAD/OULAD_processed.csv
# Full Dataset
# Student Simulator 1

df1 = pd.read_csv("/data/aimotion/OULAD/OULAD_processed.csv")
df1['finalgrade'] = df1['norm_finalgrade']*100
df1 = df1.drop('Unnamed: 0', axis=1)
df_trimmed1 = df1[df1['finalgrade'].between(0.001, 99.999)]

def StudentSimulator(df, size, add_method="reuse", userid = 'userid', seed=123):
    import random
    #Create the df
    data = pd.DataFrame(columns=['userid', 'itemid', 'norm_finalgrade', 'time_diff', 'timemodified', 'courseid', 'finalgrade'])
    
    np.random.seed(seed)
    vector = np.vectorize(np.int_)
    
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
       
    elif add_method == "simulate":
        #Adding userid accoring to statistics (mean=7, std=10)
        df = UserReindex(df = df, userid = userid)
        vector = np.vectorize(np.int_)
        uniques = int(size*df['userid_n'].nunique()/len(df))
        sampler = vector(np.linspace(start=1, stop=uniques, num=uniques))

        users = df['userid_n'].value_counts().rename_axis('userid').reset_index(name='counts')
        users_count = users['counts'].value_counts().rename_axis('no_of_grades').reset_index(name='no_of_students') 
        users_count = users_count.sort_values('no_of_grades')

        prob = stats.pareto.rvs(size=uniques, b=1.86, loc=-6.788, scale=7.788, random_state=seed)
        prob /= np.sum(prob)

        sampled_users = pd.Series(np.random.choice(sampler, size, p=prob))    
        data['userid'] = sampled_users


        #Adding courseid
        uniques = int(size*df['course'].nunique()/len(df))
        sampler = vector(np.linspace(start=1, stop=uniques, num=uniques))

        courses = df['courseid'].value_counts().rename_axis('courseid').reset_index(name='counts')
        courses_count = courses['counts'].value_counts().rename_axis('no_of_grades').reset_index(name='no_of_courses') 
        courses_count = courses_count.sort_values('no_of_grades')

        prob = stats.expon.rvs(size=uniques, loc=1, scale=58.928, random_state=seed)
        prob /= np.sum(prob)

        sampled_courses = pd.Series(np.random.choice(sampler, size, p=prob))    
        data['courseid'] = sampled_courses    

        #Adding items

        items = list(range(1, size+1))
        sampled_items = []
        data = data.sort_values('courseid')
        for i in tqdm(data['courseid'].unique(), desc='Finding quizzes'):
            myitems = [items.pop(random.randrange(len(items))) for _ in range(int(max(1, 0.15*len(data[data['courseid']==i]))))]
            sampled_items.append([random.choice(myitems) for _ in range(len(data[data['courseid']==i]))])


        data['itemid'] = sum(sampled_items, [])  

    

    #Adding grades
    grade=stats.beta.rvs(a=17.2982, b=0.889271, loc=-318.198, scale=418.198, size=size)
    data['finalgrade'] = grade
    data['finalgrade'].where(data['finalgrade'] >= 0, 0, inplace=True)
    data['finalgrade'].where(data['finalgrade'] < 100, 100, inplace=True)

    data['norm_finalgrade'] = data['finalgrade']/100

    #Adding time
    time=stats.beta.rvs(a=0.92062, b=0.972657, loc=12, scale=249.03, size=size)
    data['time_diff'] = time

    return(data)

simdata = StudentSimulator(df1, len(df1))
simdata.to_csv('/data/aimotion/OULAD/StuSi_OULAD_full_full.csv')
simdata = StudentSimulator(df1, 2*len(df1))
simdata.to_csv('/data/aimotion/OULAD/StuSi_OULAD_full_double.csv')
simdata = StudentSimulator(df1, 3*len(df1))
simdata.to_csv('/data/aimotion/OULAD/StuSi_OULAD_full_triple.csv')


##############################


df2 = pd.read_csv("SLP_unit_processed_ogGrades.csv")
df2['finalgrade'] = (df2['full_score']-df2['score'])/df2['full_score']*100
df2 = df2.drop('Unnamed: 0', axis=1)
df_trimmed2 = df2[df2['finalgrade'].between(0.001, 99.999)]


# Dataset: /data/aimotion/SLP/SLP_unit_processed.csv
# Trimmed Dataset
# Student Simulator 1

def StudentSimulator(df, size, add_method="reuse", userid = 'userid', seed=123):
    import random
    #Create the df
    data = pd.DataFrame(columns=['userid', 'itemid', 'norm_finalgrade', 'time_diff', 'timemodified', 'courseid', 'finalgrade'])
    
    np.random.seed(seed)
    vector = np.vectorize(np.int_)
    
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
        time_diff = newdf1['time_diff'].to_list()
        time_modified = newdf1['timemodified'].to_list()
        data['userid'] = users
        data['courseid'] = courses
        data['itemid'] = items
        data['time_diff'] = time_diff
        data['timemodified'] = time_modified
       
    elif add_method == "simulate":
        #Adding userid accoring to statistics (mean=7, std=10)
        df = UserReindex(df = df, userid = userid)
        vector = np.vectorize(np.int_)
        uniques = int(size*df['userid_n'].nunique()/len(df))
        sampler = vector(np.linspace(start=1, stop=uniques, num=uniques))

        users = df['userid_n'].value_counts().rename_axis('userid').reset_index(name='counts')
        users_count = users['counts'].value_counts().rename_axis('no_of_grades').reset_index(name='no_of_students') 
        users_count = users_count.sort_values('no_of_grades')

        prob = stats.pareto.rvs(size=uniques, b=1.86, loc=-6.788, scale=7.788, random_state=seed)
        prob /= np.sum(prob)

        sampled_users = pd.Series(np.random.choice(sampler, size, p=prob))    
        data['userid'] = sampled_users


        #Adding courseid
        uniques = int(size*df['course'].nunique()/len(df))
        sampler = vector(np.linspace(start=1, stop=uniques, num=uniques))

        courses = df['courseid'].value_counts().rename_axis('courseid').reset_index(name='counts')
        courses_count = courses['counts'].value_counts().rename_axis('no_of_grades').reset_index(name='no_of_courses') 
        courses_count = courses_count.sort_values('no_of_grades')

        prob = stats.expon.rvs(size=uniques, loc=1, scale=58.928, random_state=seed)
        prob /= np.sum(prob)

        sampled_courses = pd.Series(np.random.choice(sampler, size, p=prob))    
        data['courseid'] = sampled_courses    

        #Adding items

        items = list(range(1, size+1))
        sampled_items = []
        data = data.sort_values('courseid')
        for i in tqdm(data['courseid'].unique(), desc='Finding quizzes'):
            myitems = [items.pop(random.randrange(len(items))) for _ in range(int(max(1, 0.15*len(data[data['courseid']==i]))))]
            sampled_items.append([random.choice(myitems) for _ in range(len(data[data['courseid']==i]))])


        data['itemid'] = sum(sampled_items, [])  

    

    #Adding grades
    grade=stats.loggamma.rvs(c=0.502433, loc=89.017, scale=8.19915, size=size)
    data['finalgrade'] = grade
    data['finalgrade'].where(data['finalgrade'] >= 0, 0, inplace=True)
    data['finalgrade'].where(data['finalgrade'] < 100, 100, inplace=True)

    data['norm_finalgrade'] = data['finalgrade']/100

    #Adding time
    #time=stats.uniform.rvs(loc=12, scale=249, size=size)
    #data['time_diff'] = time

    return(data)

simdata = StudentSimulator(df_trimmed2, len(df2))
simdata.to_csv('/data/aimotion/SLP/StuSi_SLP_trimmed_full.csv')
simdata = StudentSimulator(df_trimmed2, 2*len(df2))
simdata.to_csv('/data/aimotion/SLP/StuSi_SLP_trimmed_double.csv')
simdata = StudentSimulator(df_trimmed2, 3*len(df2))
simdata.to_csv('/data/aimotion/SLP/StuSi_SLP_trimmed_triple.csv')


# Dataset: /data/aimotion/SLP/SLP_unit_processed.csv
# Full Dataset
# Student Simulator 1

def StudentSimulator(df, size, add_method="reuse", userid = 'userid', seed=123):
    import random
    #Create the df
    data = pd.DataFrame(columns=['userid', 'itemid', 'norm_finalgrade', 'time_diff', 'timemodified', 'courseid', 'finalgrade'])
    
    np.random.seed(seed)
    vector = np.vectorize(np.int_)
    
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
        time_diff = newdf1['time_diff'].to_list()
        time_modified = newdf1['timemodified'].to_list()
        data['userid'] = users
        data['courseid'] = courses
        data['itemid'] = items
        data['time_diff'] = time_diff
        data['timemodified'] = time_modified
       
    elif add_method == "simulate":
        #Adding userid accoring to statistics (mean=7, std=10)
        df = UserReindex(df = df, userid = userid)
        vector = np.vectorize(np.int_)
        uniques = int(size*df['userid_n'].nunique()/len(df))
        sampler = vector(np.linspace(start=1, stop=uniques, num=uniques))

        users = df['userid_n'].value_counts().rename_axis('userid').reset_index(name='counts')
        users_count = users['counts'].value_counts().rename_axis('no_of_grades').reset_index(name='no_of_students') 
        users_count = users_count.sort_values('no_of_grades')

        prob = stats.pareto.rvs(size=uniques, b=1.86, loc=-6.788, scale=7.788, random_state=seed)
        prob /= np.sum(prob)

        sampled_users = pd.Series(np.random.choice(sampler, size, p=prob))    
        data['userid'] = sampled_users


        #Adding courseid
        uniques = int(size*df['course'].nunique()/len(df))
        sampler = vector(np.linspace(start=1, stop=uniques, num=uniques))

        courses = df['courseid'].value_counts().rename_axis('courseid').reset_index(name='counts')
        courses_count = courses['counts'].value_counts().rename_axis('no_of_grades').reset_index(name='no_of_courses') 
        courses_count = courses_count.sort_values('no_of_grades')

        prob = stats.expon.rvs(size=uniques, loc=1, scale=58.928, random_state=seed)
        prob /= np.sum(prob)

        sampled_courses = pd.Series(np.random.choice(sampler, size, p=prob))    
        data['courseid'] = sampled_courses    

        #Adding items

        items = list(range(1, size+1))
        sampled_items = []
        data = data.sort_values('courseid')
        for i in tqdm(data['courseid'].unique(), desc='Finding quizzes'):
            myitems = [items.pop(random.randrange(len(items))) for _ in range(int(max(1, 0.15*len(data[data['courseid']==i]))))]
            sampled_items.append([random.choice(myitems) for _ in range(len(data[data['courseid']==i]))])


        data['itemid'] = sum(sampled_items, [])  

    

    #Adding grades
    grade=stats.chi2.rvs(df=0.421997, loc=-2.25572e-25, scale=6.74993, size=size)
    data['finalgrade'] = grade
    data['finalgrade'].where(data['finalgrade'] >= 0, 0, inplace=True)
    data['finalgrade'].where(data['finalgrade'] < 100, 100, inplace=True)

    data['norm_finalgrade'] = data['finalgrade']/100

    #Adding time
    #time=stats.uniform.rvs(loc=12, scale=249, size=size)
    #data['time_diff'] = time

    return(data)

simdata = StudentSimulator(df2, len(df2))
simdata.to_csv('/data/aimotion/SLP/StuSi_SLP_full_full.csv')
simdata = StudentSimulator(df2, 2*len(df2))
simdata.to_csv('/data/aimotion/SLP/StuSi_SLP_full_double.csv')
simdata = StudentSimulator(df2, 3*len(df2))
simdata.to_csv('/data/aimotion/SLP/StuSi_SLP_full_triple.csv')
