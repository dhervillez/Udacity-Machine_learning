#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

#%%

### Load the dictionary containing the dataset
print 'Load the dictionary containing the dataset'
print ''

with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

print 'Remove some entries from the dataset'
print '\tTOTAL'
print '\tTHE TRAVEL AGENCY IN THE PARK'
print '\tLOCKHART EUGENE E'
print ''

# not individual
data_dict.pop('TOTAL', None)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', None)
# no data
data_dict.pop('LOCKHART EUGENE E', None)


names_list = sorted(data_dict.keys())

print 'Correct financial data for'
print '\tBELFER ROBERT'
print '\tBHATNAGAR SANJAY'
print ''

# correct financial data for BELFER ROBERT and BHATNAGAR SANJAY
data_dict['BELFER ROBERT']['salary']=np.nan
data_dict['BELFER ROBERT']['bonus']=np.nan
data_dict['BELFER ROBERT']['long_term_incentive']=np.nan
data_dict['BELFER ROBERT']['deferred_income']=-102500
data_dict['BELFER ROBERT']['deferral_payments']=np.nan
data_dict['BELFER ROBERT']['loan_advances']=np.nan
data_dict['BELFER ROBERT']['other']=np.nan
data_dict['BELFER ROBERT']['expenses']=3285
data_dict['BELFER ROBERT']['director_fees']=102500
data_dict['BELFER ROBERT']['total_payments']=3285
data_dict['BELFER ROBERT']['exercised_stock_options']=np.nan
data_dict['BELFER ROBERT']['restricted_stock']=44093
data_dict['BELFER ROBERT']['restricted_stock_deferred']=-44093
data_dict['BELFER ROBERT']['total_stock_value']=np.nan

data_dict['BHATNAGAR SANJAY']['salary']=np.nan
data_dict['BHATNAGAR SANJAY']['bonus']=np.nan
data_dict['BHATNAGAR SANJAY']['long_term_incentive']=np.nan
data_dict['BHATNAGAR SANJAY']['deferred_income']=-np.nan
data_dict['BHATNAGAR SANJAY']['deferral_payments']=np.nan
data_dict['BHATNAGAR SANJAY']['loan_advances']=np.nan
data_dict['BHATNAGAR SANJAY']['other']=np.nan
data_dict['BHATNAGAR SANJAY']['expenses']=137854
data_dict['BHATNAGAR SANJAY']['director_fees']=np.nan
data_dict['BHATNAGAR SANJAY']['total_payments']=137854
data_dict['BHATNAGAR SANJAY']['exercised_stock_options']=15456290
data_dict['BHATNAGAR SANJAY']['restricted_stock']=2604490
data_dict['BHATNAGAR SANJAY']['restricted_stock_deferred']=-2604490
data_dict['BHATNAGAR SANJAY']['total_stock_value']=15456290

print 'Compute new features'
print '\tratio_salary = salary/total_payments'
print '\tratio_stock_option = total_stock_value/total_payments'
print '\tratio_long_term_incentive = long_term_incentive/total_payments'
print '\tratio_to_poi = from_this_person_to_poi/from_messages'
print '\tratio_from_poi = from_poi_to_this_person/to_messages'
print ''

# compute new features
for user in data_dict:
    data_dict[user]['ratio_salary']=str(float(data_dict[user]['salary'])/float(data_dict[user]['total_payments']))
    data_dict[user]['ratio_stock_option']=str(float(data_dict[user]['total_stock_value'])/float(data_dict[user]['total_payments']))
    data_dict[user]['ratio_long_term_incentive']=str(float(data_dict[user]['long_term_incentive'])/float(data_dict[user]['total_payments']))
    data_dict[user]['ratio_to_poi']=str(float(data_dict[user]['from_this_person_to_poi'])/float(data_dict[user]['from_messages']))
    data_dict[user]['ratio_from_poi']=str(float(data_dict[user]['from_poi_to_this_person'])/float(data_dict[user]['to_messages']))

### Extract features and labels from dataset for local testing
print 'Extract features and labels from dataset'
print ''

features_list=['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus','restricted_stock_deferred',
              'deferred_income', 'total_stock_value', 'expenses','exercised_stock_options', 'other', 'long_term_incentive',
              'restricted_stock','director_fees', 'ratio_salary', 'ratio_stock_option', 'ratio_long_term_incentive', 'to_messages', 'from_poi_to_this_person', 'from_messages',
              'from_this_person_to_poi', 'shared_receipt_with_poi', 'ratio_to_poi', 'ratio_from_poi']

data = featureFormat(data_dict, features_list, sort_keys = True, remove_NaN=False)
labels, features = targetFeatureSplit(data)

#%%
# scatter plot with null ratio analysis

df = pd.DataFrame(data,columns=features_list)

df['name']=names_list

def count_na_col(df):
    return float(df.isna().sum())/len(df)

null_ratio_col= df.groupby(by='poi').agg(count_na_col) 
 
plot_colors = ['blue', 'red']
plot_labels = ['non-POI', 'POI']

print 'Create scatter plots with row data'
print '\t\t"scatter_plot_row_data.pdf"'
print ''
pp = PdfPages('scatter_plot_row_data.pdf')

# loop through features except poi and name
for feature in df.columns.values:
    if feature<>'poi' and feature<>'name':
        
        # create figure and ax with given size to fit the legend below
        fig = plt.figure()
        ax = fig.add_axes([0.2, 0.25, 0.7, 0.65])
        
        # plot the data for poi and non-poi
        for name, group in df.groupby(by='poi'):
            group.reset_index().plot(kind='scatter',x='index', y=feature, c=plot_colors[int(name)], label=plot_labels[int(name)], ax=ax, grid=True)
            
        # display null ratio for poi and non poi population
        text='''
        proportion of null in non poi population: {0:.0%}
        proportion of null in poi population: {1:.0%}
        '''.format(null_ratio_col[feature][0],null_ratio_col[feature][1])
        
        plt.figtext(0.2, 0.0, text, horizontalalignment='left')
        
        # print name of min and max for each feature
        plt.text(df[feature].idxmax(),df[feature].max(),df['name'].iloc[df[feature].idxmax()])
        plt.text(df[feature].idxmin(),df[feature].min(),df['name'].iloc[df[feature].idxmin()])
        
        # save figure in pdf
        pp.savefig()
        plt.close()

pp.close()   

#%%

import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.impute import SimpleImputer

# define imputer and scaler to prepare the data before machine learning

financial_features = np.arange(0,17)
email_features = np.arange(17,24)

imputer = ColumnTransformer(
        [('financial', SimpleImputer(missing_values=np.nan, strategy='constant'), financial_features),
        ('email', SimpleImputer(missing_values=np.nan, strategy='median'), email_features)])
scaler = PowerTransformer()

# histogram plots with row data

print 'Create plots for visual data analysis'
print '\tHistogram plots with row data'
print '\t\t"hist_row_data.pdf"'
pp = PdfPages('hist_row_data.pdf')
df = pd.DataFrame(data,columns=features_list)
for feature in df.columns.values:
    if feature<>'poi' and feature<>'name':
        # create figure and ax with given size to fit the legend below
        fig = plt.figure()
        ax = fig.add_axes([0.2, 0.25, 0.7, 0.65])
        plt.xlabel(feature)
        for name, group in df.groupby(by='poi'):
            group.plot(kind='hist',y=feature, color=plot_colors[int(name)], label=plot_labels[int(name)], ax=ax, grid=True)
        pp.savefig()
        plt.close()
pp.close()          

# histogram plots with row data after imputer and after scaling

print '\tHistogram plots with row data after imputer'
print '\t\t"hist_after_imputer.pdf"'
pp = PdfPages('hist_after_imputer.pdf')
features_t=imputer.fit_transform(np.array(features))
df = pd.DataFrame(np.concatenate((np.array(labels,ndmin=2).T,features_t),axis=1),columns=features_list)
for feature in df.columns.values:
    if feature<>'poi' and feature<>'name':
        # create figure and ax with given size to fit the legend below
        fig = plt.figure()
        ax = fig.add_axes([0.2, 0.25, 0.7, 0.65])
        plt.xlabel(feature)
        for name, group in df.groupby(by='poi'):
            group.plot(kind='hist',y=feature, color=plot_colors[int(name)], label=plot_labels[int(name)], ax=ax, grid=True)
        pp.savefig()
        plt.close()
pp.close() 

# histogram plots with row data after imputer and scaling

print '\tHistogram plots with row data after imputer and scaler'
print '\t\t"hist_after_scaler.pdf"'
print ''
pp = PdfPages('hist_after_scaler.pdf')
pipe = Pipeline(steps=[
        ('imputer',imputer),
        ('scaler',scaler)
        ])
features_t=pipe.fit_transform(np.array(features))
df = pd.DataFrame(np.concatenate((np.array(labels,ndmin=2).T,features_t),axis=1),columns=features_list)
for feature in df.columns.values:
    if feature<>'poi' and feature<>'name':
        # create figure and ax with given size to fit the legend below
        fig = plt.figure()
        ax = fig.add_axes([0.2, 0.25, 0.7, 0.65])
        plt.xlabel(feature)
        for name, group in df.groupby(by='poi'):
            group.plot(kind='hist',y=feature, color=plot_colors[int(name)], label=plot_labels[int(name)], ax=ax, grid=True)
        pp.savefig()
        plt.close()
pp.close()             



#%%
# ranking by features importance thanks to random_forest

from sklearn.ensemble import RandomForestClassifier

print 'Compute feature importances with random forest algorithm'
print ''

forest = RandomForestClassifier(n_estimators=100, random_state = 42)

pipe = Pipeline(steps=[
        ('imputer',imputer),
        ('scaler',scaler),
        ('classifier', forest)
        ])

pipe.fit(np.array(features), np.array(labels))
importances = pipe.named_steps['classifier'].feature_importances_
std = np.std([tree.feature_importances_ for tree in pipe.named_steps['classifier'].estimators_],axis=0)
indices = np.argsort(importances)[::-1]
indices_label = [features_list[1:][i] for i in indices]

# Print the feature ranking
print("Feature ranking:")

for f in range(np.array(features).shape[1]):
    print("{0:.0f} {1} ({2:.2%})".format(f + 1, indices_label[f], importances[indices[f]]))
print ''

# Plot the feature importances of the random forest
print 'Create histogram plots with feature importances based on random forest algorithm'
print '\t\t"random_forest_features_importance.pdf"'
print ''
pp = PdfPages('random_forest_features_importance.pdf')
fig = plt.figure()
ax = fig.add_axes([0.1, 0.4, 0.8, 0.5])
plt.title("Features importance based on decision tree algorithm", fontsize=10)
ax.bar(range(np.array(features).shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center", zorder=2)
plt.grid(zorder=1)
plt.xticks(range(np.array(features).shape[1]), indices_label, rotation='vertical', fontsize=6)
plt.xlim([-1, np.array(features).shape[1]])
plt.yticks(np.arange(-0.05,0.25,0.05), fontsize=6)
plt.ylim(-0.05,0.2)
pp.savefig()
plt.close()
pp.close()

