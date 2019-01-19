#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import numpy as np

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list=['poi', 'exercised_stock_options' , 'other', 'bonus', 'expenses',
               'ratio_to_poi']

#%%
### Load the dictionary containing the dataset
print 'Load the dictionary containing the dataset'
print ''
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
### Task 2: Remove outliers
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

#%%

### Task 3: Create new feature(s)

print 'Compute new features'
print '\tratio_to_poi = from_this_person_to_poi/from_messages'
print ''

for user in data_dict:
    data_dict[user]['ratio_to_poi']=str(float(data_dict[user]['from_this_person_to_poi'])/float(data_dict[user]['from_messages']))


### Store to my_dataset for easy export below.
my_dataset = data_dict

print 'Extract features and labels from dataset'
print ''

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True, remove_NaN=False)
labels, features = targetFeatureSplit(data)
#%%


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html




### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import json

from sklearn.model_selection import  StratifiedShuffleSplit   
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix

def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]

PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\
\tFalse negatives: {:4d}\tTrue negatives: {:4d}"

# impute missing values (nan in dataset) with 0 for financial data and the median of the population for email data
# columns transformer is used. list of index array are provided to each impute function
financial_features = np.arange(0,4)
email_features = np.arange(4,5)

imputer = ColumnTransformer(
        [('financial', SimpleImputer(missing_values=np.nan, strategy='constant'), financial_features),
        ('email', SimpleImputer(missing_values=np.nan, strategy='median'), email_features)])

# scaling is used to improve performance of some algorithm (SVC in particualar)
# robust scaler, which better handle outliers than standard scaler, showed better performance
scaler = PowerTransformer()

# several classifiers are tested
# for each classifier, hyperparameters are tuned with gridsearchcv
# for each classifier, performance is evaluated with cross validation

classifier_list=['NB', 'SVC', 'KNN','random_forest', 'AdaBoost']

#%%
print 'Test several classifiers with GridSearchCV'
print ''

file = open('GridSearchCV_resutls_5_features.txt','w')

file.write('Identify fraud from Enron email and financial information\n')
file.write('\n')
file.write('List of selected features\n')
file.write('\t'+json.dumps(features_list)+'\n')
file.write('\n')
file.write('List of individuals in the dataset\n')
file.write('\t'+json.dumps(names_list)+'\n')
file.write('\n')
n_POI = int(np.array(labels).sum())
n_non_POI = int(len(labels)) - n_POI
file.write('Cleaned dataset has {0:d} POI and {1:d} non POI\n'.format(n_POI, n_non_POI))
file.write('\n')


for classifier_name in classifier_list:
    
    print classifier_name
    file.write('Classifier name: {0}\n'.format(classifier_name))
    
    # hyperparameter setting
    parameters = {}
    
    if classifier_name == 'NB':
        # Naive Bayes --> no hyperparameter to tune
        classifier = GaussianNB()        

    if classifier_name == 'SVC':
        # Support vector machine--> kernel and margin are tuned
        classifier = SVC()
        parameters['classifier__kernel'] = ['linear', 'poly', 'rbf'] 
        parameters['classifier__C'] = [10,100,1000]
        parameters['classifier__gamma'] = [0.01,0.1,1,'scale']

    if classifier_name == 'KNN':
        # KNN --> number of neighbors, weight function and power parameters are tuned
        classifier = KNeighborsClassifier(algorithm='auto')
        parameters['classifier__n_neighbors'] = [5,10,15]
        parameters['classifier__weights'] = ['distance', 'uniform']
        parameters['classifier__p'] = [1,2]

    if classifier_name == 'random_forest':
        # random forest --> number of trees, min sample split and min sample leaf are tuned
        classifier = RandomForestClassifier(random_state=42)
        parameters['classifier__min_samples_split'] = [2,4,6]
        parameters['classifier__min_samples_leaf'] = [1,2,4]
        parameters['classifier__n_estimators'] = [10,20,40]

    if classifier_name == 'AdaBoost':
        # adaboost --> number of estimators is tuned
        classifier = AdaBoostClassifier(random_state=42)
        parameters['classifier__n_estimators'] = [10,20,40]

    # three steps of the algorithm are gathered in a pipeline
    pipe = Pipeline(steps=[
            ('imputer',imputer),
            ('scaler',scaler),
            ('classifier', classifier)
             ])
    
    # hyperparameter tuning is performed with gridsearchcv
    
    # cross validation with stratifed shuffle split strategy
    cv = StratifiedShuffleSplit(n_splits=100, test_size=0.1, random_state = 42)
    
    # best parameters are selected such as to maximize the recall score
    scoring='f1'
    
    clf = GridSearchCV(pipe, parameters, cv=cv, scoring=scoring, error_score=np.nan, verbose=0, n_jobs=-1)
    
    clf.fit(np.array(features), np.array(labels))
    
    # print  the main results of GridSearchCV
    print 'GridSearchCV results\n'
    print 'Grid of hyper parameters\n'
    print '\t'+json.dumps(parameters)+'\n'
    print 'Best hyper parameters\n'
    print '\t'+json.dumps(clf.best_params_)+'\n'
    print 'Best {0} score: {1:.0%}\n'.format(scoring, clf.best_score_)

    file.write('GridSearchCV results\n')
    file.write('\tGrid of hyperparameters\n')
    file.write('\t'+json.dumps(parameters)+'\n')
    file.write('\tBest hyperparameters\n')
    file.write('\t'+json.dumps(clf.best_params_)+'\n')
    file.write('\tBest {0} score: {1:.0%}\n'.format(scoring, clf.best_score_))

    # models are then evaluated with the best hyperparameters thanks to cross validation
    if classifier_name == 'NB':
        classifier = GaussianNB()

    if classifier_name == 'SVC':
        classifier = SVC(kernel=clf.best_params_['classifier__kernel'],
                         C=clf.best_params_['classifier__C'],
                         gamma=clf.best_params_['classifier__gamma'])
    
    if classifier_name == 'KNN':
        classifier = KNeighborsClassifier(n_neighbors=clf.best_params_['classifier__n_neighbors'],
                                          weights=clf.best_params_['classifier__weights'],
                                          algorithm='auto',
                                          p=clf.best_params_['classifier__p'])
    if classifier_name == 'random_forest':
        classifier = RandomForestClassifier(min_samples_split=clf.best_params_['classifier__min_samples_split'],
                                            min_samples_leaf=clf.best_params_['classifier__min_samples_leaf'],
                                            n_estimators=clf.best_params_['classifier__n_estimators'])
    if classifier_name == 'AdaBoost':
        classifier = AdaBoostClassifier(n_estimators=clf.best_params_['classifier__n_estimators'], random_state=42)

    # three steps of the algorithm are gathered in a pipeline
    clf = Pipeline(steps=[
            ('imputer',imputer),
            ('scaler',scaler),
            ('classifier', classifier)
             ])

    # cross validation with stratifed shuffle split strategy
    cv = StratifiedShuffleSplit(n_splits=1000, test_size=0.1, random_state = 42)

    # scoring dict allows to widthdraw each element of the confusion matrix
    # tp --> true positive
    # tn --> true negative
    # fp --> false positive
    # fn --> false negative
    scoring = {'tp': make_scorer(tp), 'tn': make_scorer(tn),
               'fp': make_scorer(fp), 'fn': make_scorer(fn)}
    
    scores = cross_validate(clf, np.array(features), np.array(labels), scoring=scoring, cv=cv, return_train_score=False)

    true_positives=scores['test_tp'].sum()
    false_positives=scores['test_fp'].sum()
    false_negatives=scores['test_fn'].sum()
    true_negatives=scores['test_tn'].sum()

    # metrics on the overall testing results are computed based on confusion matrix results
    total_predictions = true_negatives + false_negatives + false_positives + true_positives
    if (true_positives + true_negatives)==0:
        accuracy = np.nan
    else:
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        
    if true_positives==0:
        precision = np.nan
        recall = np.nan
        f1 = np.nan
        f2 = np.nan
    else:
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)

    # print  the main results of cross validation
    print 'Cross validation results'
    print PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5)
    print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives)
    print ''
    
    file.write('Cross validation results\n')
    file.write(PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5)+'\n')
    file.write(RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives)+'\n')
    file.write(' \n')
    
file.close() 

#%%

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

print 'Dump best classifier, dataset, and features_list so that resutls can be checked with poi_id.py'

clf = Pipeline(steps=[
        ('imputer',imputer),
        ('scaler',scaler),
        ('classifier', GaussianNB())
        ])

dump_classifier_and_data(clf, my_dataset, features_list)



