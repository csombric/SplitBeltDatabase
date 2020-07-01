
# coding: utf-8

# Choices that I am making when modeling:
# - Features to use
# - How I split the data (Unclear how I ought to do this... I guess I should look through the repo that Rayid shared)
# - Smote
# - Models to run 
# - Constraints to the model
# 
# Metrics worth looking at 
# - ROC (Threshold independent)
# - Accuracy @ X%
# - Precisiton @ X%
# - Recall @ X%
# 
# https://github.com/dssg/triage/blob/master/src/triage/experiments/model_grid_presets.yaml

# Include standard modules
import argparse

# Initiate the parser
parser = argparse.ArgumentParser()
parser.add_argument('-l','--list', nargs='+', help='<Required> Set flag', required=True)
args = parser.parse_args()

#print(args.list[2])

# The following will be an input to this script
c_list = [args.list[0]]
NumberOfFeatures = int(args.list[1])
UseTransformedData = int(args.list[2])
target_cols = args.list[3]

#print(type(UseTransformedData))

import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
from patsy import dmatrices

# Use the user specified data
# Use the user specified data
if UseTransformedData == 0:
    df = pd.read_csv("CleanDataBase.csv")
    WhichDataBase = ["UnNormalized"]
    
    Y, X = dmatrices(target_cols + ' ~ C(SpeedRatio) + SpeedDifference + MidSpeed+ C(Abrupt) +\
    TMBase + AdaptationDuration + Age + C(Young) + Height + Weight + BMI +\
    C(IsCatch) + C(Stroke)', df, return_type="dataframe")

    feature_cols = ['C(SpeedRatio)[T.3.0]', 'SpeedDifference', 'MidSpeed', 'C(Abrupt)[T.1]', \
    'TMBase', 'AdaptationDuration', 'Age', 'C(Young)[T.1]', 'Height', 'Weight',\
    'BMI', 'C(IsCatch)[T.1.0]', 'C(Stroke)[T.1]']

elif UseTransformedData == 1:
    df = pd.read_csv("TransformedScaledDataBase.csv")
    WhichDataBase = ["Transformed"]
    
    Y, X = dmatrices(target_cols + ' ~ C(SpeedRatio) + SpeedDifference + MidSpeed+ C(Abrupt) +\
    TMBase + AdaptationDuration + Age + C(Young) + Height + Weight + BMI +\
    C(IsCatch) + C(Stroke)', df, return_type="dataframe")

    feature_cols = ['C(SpeedRatio)[T.3.0]', 'SpeedDifference', 'MidSpeed',\
    'C(Abrupt)[T.1.0]', 'TMBase', 'AdaptationDuration', 'Age', \
    'C(Young)[T.1.0]', 'Height', 'Weight',\
    'BMI', 'C(IsCatch)[T.1.0]', 'C(Stroke)[T.1.0]']
# In[4]:


#Define possible features and target


df_Interactions = pd.concat([X,Y], axis=1)

Features = df_Interactions[feature_cols]
Target = df_Interactions[target_cols]


# In[4]:


# Define Model Specific Variations
DT = {
    'max_depth': [1,5,10],
    'max_features': ['sqrt','log2'],
    'min_samples_split': [2, 5, 10],
    }

RF = {
    'n_estimators': [10,100],
    'max_depth': [1,5,10],
    'max_features': ['sqrt','log2'],
    'min_samples_split': [2, 5, 10],
    }

ET = {
    'n_estimators': [10,100],
    'max_depth': [1,5,10],
    'max_features': ['sqrt','log2'],
    'min_samples_split': [2, 5, 10],
    }

LR = {
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'C': [0.00001,0.0001,0.001,0.01,0.1,1,10],
    'max_iter': [10000], 
    } 


# In[5]:


def generate_binary_at_k(y_scores, k):
    cutoff_index = int(len(y_scores) * (k / 100.0))
    predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
    return predictions_binary

def joint_sort_descending(l1, l2):
    # l1 and l2 have to be numpy arrays
    idx = np.argsort(l1)[::-1]
    return l1[idx], l2[idx]

def precision_at_k(y_true, y_scores, k):
    #y_scores_sorted, y_true_sorted = zip(*sorted(zip(y_scores, y_true), reverse=True))
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    #precision, _, _, _ = metrics.precision_recall_fscore_support(y_true, preds_at_k)
    #precision = precision[1]  # only interested in precision for label 1
    precision = precision_score(y_true_sorted, preds_at_k)
    return precision

def recall_at_k(y_true, y_scores, k):
    #y_scores_sorted, y_true_sorted = zip(*sorted(zip(y_scores, y_true), reverse=True))
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    #precision, _, _, _ = metrics.precision_recall_fscore_support(y_true, preds_at_k)
    #precision = precision[1]  # only interested in precision for label 1
    recall = recall_score(y_true_sorted, preds_at_k)
    return recall


# In[6]:


import itertools
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, 
AdaBoostClassifier)
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, accuracy_score, precision_recall_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, SGDClassifier

c_MDL_list = {'DT': DecisionTreeClassifier, 
              'RF': RandomForestClassifier, 
              'ET': ExtraTreesClassifier,
              'GB': GradientBoostingClassifier,
              'LR': LogisticRegression,
              'SGD': SGDClassifier, 
              'KNN': KNeighborsClassifier,
              'SVC': SVC,
              'GPC': GaussianProcessClassifier,
              'MLP': MLPClassifier,
              'ABC': AdaBoostClassifier}

def RunThroughModelInputs(ModelResults, X_train, y_train, X_test, y_test, FeaturesUsed, SplitUsed, WhichDataBase):   
    
    for c_index in c_list:
        #print(c_index)
        vectors = eval(c_index + '.values()')
        keys = eval(c_index + '.keys()')
        #vectors = RF.values()  # list of lists
        #keys = RF.keys()
        
        model_name = c_MDL_list[c_index]

        for x in itertools.product(*vectors): # itertools.product(first_item, second_item, ..., last_item)

            Inputs2MDL = dict(zip(keys,x))

            model = model_name(**Inputs2MDL)

            model.fit(X_train, y_train)

            predicted_prob = np.array(model.predict_proba(X_test)[:,1])

            roc = roc_auc_score(y_test, predicted_prob)

            ModelResults = ModelResults.append({'Classifier': c_list, 'Parameters': model,  'Split Type': SplitUsed,
                                               'Features': FeaturesUsed, "ROC": roc, "Data": WhichDataBase,
                                              'Baseline': precision_at_k(y_test,predicted_prob, 1), 
                                               'Precision_5%': precision_at_k(y_test,predicted_prob, 5),
                                               'Precision_10%': precision_at_k(y_test,predicted_prob, 10),
                                                'Precision_20%': precision_at_k(y_test,predicted_prob, 20),
                                                'Precision_30%': precision_at_k(y_test,predicted_prob, 30),
                                                'Precision_40%': precision_at_k(y_test,predicted_prob, 40),
                                               'Precision_50%': precision_at_k(y_test,predicted_prob, 50),
                                               'Recall_5%': recall_at_k(y_test,predicted_prob, 5),
                                               'Recall_10%': recall_at_k(y_test,predicted_prob, 10),
                                                'Recall_20%': recall_at_k(y_test,predicted_prob, 20),
                                                'Recall_30%': recall_at_k(y_test,predicted_prob, 30),
                                                'Recall_40%': recall_at_k(y_test,predicted_prob, 40),
                                               'Recall_50%': recall_at_k(y_test,predicted_prob, 50)}, ignore_index=True)


    return ModelResults




# In[7]:


from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold

kf = KFold(n_splits=5)
    
def SplitTheDataDifferentWays(ModelResults, Features, Target, FeaturesUsed, WhichDataBase):
    
    for train_index, test_index in kf.split(X):
        
        ##### K-FOLD
        X_train, X_test = Features.iloc[train_index], Features.iloc[test_index]
        y_train, y_test = Target.iloc[train_index], Target.iloc[test_index]
        
        y_test = y_test.values.ravel()
        y_train = y_train.values.ravel()
        
        # For all the training and testing run the models
        ModelResults = RunThroughModelInputs(ModelResults, X_train, y_train, X_test, y_test, FeaturesUsed, "KFold", WhichDataBase)
        
        ##### Also want to try SMOTING the data
        os = SMOTE(random_state=0)
        os_data_X, os_data_y = os.fit_sample(X_train, y_train)
        os_data_X = pd.DataFrame(data = os_data_X, columns =  FeaturesUsed)

        X_train = os_data_X
        y_train = os_data_y

        ModelResults = RunThroughModelInputs(ModelResults, X_train, y_train, X_test, y_test, FeaturesUsed, "KFold_SMOTE", WhichDataBase)
    
    return ModelResults


# In[ ]:


from itertools import combinations 

def FeatureSelection(df, ModelResults, feature_cols, Target, NumberOfFeatures, WhichDataBase):
    
    FeatureNum = len(feature_cols)
    FeatureOptions = np.linspace(0, FeatureNum-1, FeatureNum).astype(int)

    comb = list(combinations(FeatureOptions, NumberOfFeatures))
    NumComb = len(comb)
    
    for CombF in range(NumComb):
        
        print('On Feature Combo # '+ str(CombF+1) + ' of ' + str(NumComb))
        
        FeaturesUsed = [feature_cols[i-1] for i in comb[CombF]]
        Features = df[FeaturesUsed]

        ModelResults = SplitTheDataDifferentWays(ModelResults, Features, Target, FeaturesUsed, WhichDataBase)
    
    return ModelResults


# In[ ]:


# RUN ALL THE THINGS
import time

start = time.time()

ModelResults = pd.DataFrame()

ModelResults = FeatureSelection(df_Interactions, ModelResults, feature_cols, Target, NumberOfFeatures, WhichDataBase)


print ('Took this many minutes to run: ', ((time.time() - start)/60) )

#ModelResults


# In[11]:


timestr = time.strftime("_%Y_%m_%d")
ModelResults.to_csv('/ihome/tgelsy/cjs180/Results/ModelResults_' + c_list[0] + '_' + str(NumberOfFeatures) + 'Vars_'+ target_cols + '_' + WhichDataBase[0] + timestr + '.csv', index=False)

#ModelResults.to_csv('/ihome/tgelsy/cjs180/Results/ModelResults_' + c_list[0] + '_' + str(NumberOfFeatures) + 'Vars_'+ target_cols[0] + '_' + WhichDataBase[0] + timestr + '.csv', index=False)
