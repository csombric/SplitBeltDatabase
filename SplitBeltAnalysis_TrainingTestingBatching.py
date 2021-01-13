# SplitBeltAnalysis_TrainingTestingBatching: This script trains and tests many different models. The models are run on different splits of the data. Performance metrics for each model and each split are stored in an output file.# 

# Include standard modules
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
from patsy import dmatrices

import itertools
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, accuracy_score, precision_recall_curve
from sklearn.metrics import f1_score, cohen_kappa_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold, train_test_split

from itertools import combinations 

import time


# Initiate the parser if batching on a server
#parser = argparse.ArgumentParser()
#parser.add_argument('-l','--list', nargs='+', help='<Required> Set flag', required=True)
#args = parser.parse_args()
#c_list = [args.list[0], args.list[1], args.list[2], args.list[3]]
#target_cols = args.list[4]


# Otherwise manually define inputs
c_list = ['DT', 'RF', 'ET', 'LR', 'AB', 'GB']
#target_cols = 'AE'
target_cols = 'AE_Median'



# Read in data and split into folds based on dates.

df = pd.read_csv("SQLDataBase.csv")
WhichDataBase = ["SQL"]


df.loc[df[(df['date'] == '1916-01-01') ].index.values, 'order']= 1
df.loc[df[(df['date'] == '1917-01-01') ].index.values, 'order']= 2
df.loc[df[(df['date'] == '1918-01-01') ].index.values, 'order']= 3
df.loc[df[(df['date'] == '1919-01-01') ].index.values, 'order']= 4
df.loc[df[(df['date'] == '1920-01-01') ].index.values, 'order']= 5

# Define Model Features and Targets
# Option 1 with SpeedDiff
Y, X = dmatrices(target_cols + ' ~ speed_diff +base + duration + age +  height +  bmi + C(clinical) + C(male) + C(tmfirst) + basestd + order', df, return_type="dataframe")
print(list(X))
feature_cols = ['speed_diff', 'base', 'duration', 'age', 'height', 'bmi', 'C(clinical)[T.1.0]','C(male)[T.1.0]', 'C(tmfirst)[T.1.0]', 'basestd']

# Option 1 with ONLY SpeedDiff and TMfirst
#Y, X = dmatrices(target_cols + ' ~ speed_diff + C(tmfirst) + order', df, return_type="dataframe")
#print(list(X))
#feature_cols = ['speed_diff', 'C(tmfirst)[T.1.0]']


df_Interactions = pd.concat([X,Y], axis=1)

Features = df_Interactions[feature_cols]
Target = df_Interactions[target_cols]


# Define Model Specific Variations
DT = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [2,5,10],
    'min_samples_split': [2, 10, 50],
    'class_weight': [None, 'balanced'],
    }

RF = {
    'n_estimators': [10,100,1000,10000],
    'criterion': ['gini'],
    'max_depth': [2,5,10],
    'min_samples_split': [2, 10, 50],
    'max_features': ['auto', 'sqrt'],
    'class_weight': [None, 'balanced'],
    }

ET = {
    'n_estimators': [10,100,1000,10000],
    'criterion': ['gini'],
    'max_depth': [2,5,10],
    'min_samples_split': [2, 10, 50],
    'max_features': ['auto', 'sqrt'],
    'class_weight': [None, 'balanced'],
    }

LR = {
    'C': [0.00001,0.0001,0.001,0.01,0.1,1,10],
    'max_iter': [10000],
    'class_weight': [None, 'balanced'],
    } 

AB = {
    'algorithm': ['SAMME.R'],
    'n_estimators': [5, 10, 100, 1000], 
    'learning_rate': [0.01, 0.1, 0.5, 1],
    }

GB = {
    'n_estimators': [1, 10,100,1000],
    'learning_rate': [0.01, 0.1, 0.5, 1],
    'subsample': [0.5, 1.0],
    'min_samples_split': [2, 5, 10],
    'max_depth': [2,5,10],
    }


c_MDL_list = {'DT': DecisionTreeClassifier, 
              'RF': RandomForestClassifier, 
              'ET': ExtraTreesClassifier,
              'LR': LogisticRegression,
              'AB': AdaBoostClassifier,
              'GB': GradientBoostingClassifier,
             }



# Split Data by dates
YA21 = df_Interactions[(df_Interactions['order'] == 1) ].index.values
YA31 = df_Interactions[(df_Interactions['order'] == 2) ].index.values
YG = df_Interactions[(df_Interactions['order'] == 3) ].index.values
Old = df_Interactions[(df_Interactions['order'] == 4) ].index.values
Stroke = df_Interactions[(df_Interactions['order'] == 5) ].index.values


# Creating a Dictionary of "Training" Indexes
TrainDictIndex = {} 
  
TrainDictIndex["1"] = YA21
TrainDictIndex["2"] = np.concatenate((YA21, YA31), axis=None)
TrainDictIndex["3"] = np.concatenate((YA21, YA31, YG), axis=None)
TrainDictIndex["4"] = np.concatenate((YA21, YA31, YG, Old), axis=None)
  
# Creating a Dictionary of "Testing" Indexes
TestDictIndex = {} 
  
TestDictIndex["1"] = YA31
TestDictIndex["2"] = YG
TestDictIndex["3"] = Old
TestDictIndex["4"] = Stroke
#

# Define parameter metrics that will be called upon later

def generate_binary_at_k(y_scores, k):
    cutoff_index = int(len(y_scores) * (k / 100.0))
    predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
    return predictions_binary



def joint_sort_descending(l1, l2):
    # l1 and l2 have to be numpy arrays
    idx = np.argsort(l1)[::-1]
    return l1[idx], l2[idx]



def precision_at_k(y_true, y_scores, k):
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    precision = precision_score(y_true_sorted, preds_at_k)
    return precision



def recall_at_k(y_true, y_scores, k):
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    recall = recall_score(y_true_sorted, preds_at_k)
    return recall


def accuracy_at_k(y_true, y_scores, k):
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    accuracy = accuracy_score(y_true_sorted, preds_at_k)
    return accuracy

def f1_score_at_k(y_true, y_scores, k):
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    F1 = f1_score(y_true_sorted, preds_at_k)
    return F1

def Cohen_score_at_k(y_true, y_scores, k):
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)

    Cohen_Kappa = cohen_kappa_score(y_true_sorted, preds_at_k)
    return Cohen_Kappa


from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=0)

# Temporal Cross Validation Method
def RunThroughModelInputs(ModelResults, Features,Target, WhichDataBase):   
    
    for c_index in c_list:
        vectors = eval(c_index + '.values()')
        keys = eval(c_index + '.keys()')
        
        model_name = c_MDL_list[c_index]

        for x in itertools.product(*vectors): # itertools.product(first_item, second_item, ..., last_item)

            Inputs2MDL = dict(zip(keys,x))

            model = model_name(**Inputs2MDL)
            
            # Data Splitting
            
            tempKFoldResults = pd.DataFrame()
            
            fi = pd.DataFrame()
            
            for t in ["1", "2", "3", "4"]:
                # Temporal Adding Method
                X_train = Features.loc[TrainDictIndex[t]]
                X_test = Features.loc[TestDictIndex[t]]
                y_train = Target.loc[TrainDictIndex[t]]
                y_test = Target.loc[TestDictIndex[t]]
                y_test = y_test.values.ravel()
                y_train = y_train.values.ravel()
                
                
                #SMOTTING
                os_data_X,os_data_y=os.fit_sample(X_train, y_train)
                os_data_X = pd.DataFrame(data = os_data_X, columns = feature_cols)

                X_train = os_data_X
                y_train = os_data_y
                
                model.fit(X_train, y_train)

                predicted_prob = np.array(model.predict_proba(X_test)[:,1])

                roc = roc_auc_score(y_test, predicted_prob)
                
                if c_index != 'LR':
                    fi.insert(0, t, model.feature_importances_, True) 
                
                tempKFoldResults = tempKFoldResults.append({"AUC-ROC": roc,
                                                'CarlyBaseline': y_train.mean(),
                                              'Baseline': precision_at_k(y_test,predicted_prob, 100),
                                                'PredicedProb': predicted_prob,
                                                'F1_score_90%': f1_score_at_k(y_test,predicted_prob, 90),
                                                'F1_score_80%': f1_score_at_k(y_test,predicted_prob, 80),
                                                'F1_score_70%': f1_score_at_k(y_test,predicted_prob, 70),
                                                'F1_score_60%': f1_score_at_k(y_test,predicted_prob, 60),
                                                'F1_score_50%': f1_score_at_k(y_test,predicted_prob, 50),
                                                'Cohen_Kappa_90%': Cohen_score_at_k(y_test,predicted_prob, 90),
                                                'Cohen_Kappa_80%': Cohen_score_at_k(y_test,predicted_prob, 80),
                                                'Cohen_Kappa_70%': Cohen_score_at_k(y_test,predicted_prob, 70),
                                                'Cohen_Kappa_60%': Cohen_score_at_k(y_test,predicted_prob, 60),
                                                'Cohen_Kappa_50%': Cohen_score_at_k(y_test,predicted_prob, 50),
                                               'Precision_5%': precision_at_k(y_test,predicted_prob, 5),
                                               'Precision_10%': precision_at_k(y_test,predicted_prob, 10),
                                                'Precision_20%': precision_at_k(y_test,predicted_prob, 20),
                                                'Precision_30%': precision_at_k(y_test,predicted_prob, 30),
                                                'Precision_40%': precision_at_k(y_test,predicted_prob, 40),
                                               'Precision_50%': precision_at_k(y_test,predicted_prob, 50),
                                                'Precision_60%': precision_at_k(y_test,predicted_prob, 60),
                                                'Precision_70%': precision_at_k(y_test,predicted_prob, 70),
                                                'Precision_80%': precision_at_k(y_test,predicted_prob, 80),
                                                'Precision_90%': precision_at_k(y_test,predicted_prob, 90),
                                                'Precision_95%': precision_at_k(y_test,predicted_prob, 95),
                                               'Recall_5%': recall_at_k(y_test,predicted_prob, 5),
                                               'Recall_10%': recall_at_k(y_test,predicted_prob, 10),
                                                'Recall_20%': recall_at_k(y_test,predicted_prob, 20),
                                                'Recall_30%': recall_at_k(y_test,predicted_prob, 30),
                                                'Recall_40%': recall_at_k(y_test,predicted_prob, 40),
                                               'Recall_50%': recall_at_k(y_test,predicted_prob, 50),
                                               'Recall_60%': recall_at_k(y_test,predicted_prob, 60),
                                               'Recall_70%': recall_at_k(y_test,predicted_prob, 70),
                                               'Recall_80%': recall_at_k(y_test,predicted_prob, 80),
                                               'Recall_90%': recall_at_k(y_test,predicted_prob, 90),
                                               'Recall_95%': recall_at_k(y_test,predicted_prob, 95)},ignore_index=True)
            
            #print(tempKFoldResults["PredicedProb"])
            ModelResults = ModelResults.append({'Classifier': c_index, 'Model': model, 'Parameters': Inputs2MDL,  
                                                'Split Type': "TemporalAdding","Data": WhichDataBase,
                                                'FinalPredictedProb': tempKFoldResults.loc[3, "PredicedProb"],
                                               'Baseline': tempKFoldResults["Baseline"].mean(), 
                                                "FeatureImportance": dict(zip(feature_cols, fi.mean(axis=1))),
                                                'All_F1_score_90%': tempKFoldResults["F1_score_90%"].tolist(),
                                                'F1_score_90%': tempKFoldResults["F1_score_90%"].mean(),
                                                'F1_score_90%_Var': tempKFoldResults["F1_score_90%"].var(),
                                                'All_F1_score_80%': tempKFoldResults["F1_score_80%"].tolist(),
                                                'F1_score_80%': tempKFoldResults["F1_score_80%"].mean(),
                                                'F1_score_80%_Var': tempKFoldResults["F1_score_80%"].var(),
                                                'All_F1_score_70%': tempKFoldResults["F1_score_70%"].tolist(),
                                                'F1_score_70%': tempKFoldResults["F1_score_70%"].mean(),
                                                'F1_score_70%_Var': tempKFoldResults["F1_score_70%"].var(),
                                                'All_F1_score_60%': tempKFoldResults["F1_score_60%"].tolist(),
                                                'F1_score_60%': tempKFoldResults["F1_score_60%"].mean(),
                                                'F1_score_60%_Var': tempKFoldResults["F1_score_60%"].var(),
                                                'All_F1_score_50%': tempKFoldResults["F1_score_50%"].tolist(),
                                                'F1_score_50%': tempKFoldResults["F1_score_50%"].mean(),
                                                'F1_score_50%_Var': tempKFoldResults["F1_score_50%"].var(),
                                                'All_Cohen_Kappa_90%': tempKFoldResults["Cohen_Kappa_90%"].tolist(),
                                                'Cohen_Kappa_90%': tempKFoldResults["Cohen_Kappa_90%"].mean(),
                                                'Cohen_Kappa_90%_Var': tempKFoldResults["Cohen_Kappa_90%"].var(),
                                                'All_Cohen_Kappa_80%': tempKFoldResults["Cohen_Kappa_80%"].tolist(),
                                                'Cohen_Kappa_80%': tempKFoldResults["Cohen_Kappa_80%"].mean(),
                                                'Cohen_Kappa_80%_Var': tempKFoldResults["Cohen_Kappa_80%"].var(),
                                                'All_Cohen_Kappa_70%': tempKFoldResults["Cohen_Kappa_70%"].tolist(),
                                                'Cohen_Kappa_70%': tempKFoldResults["Cohen_Kappa_70%"].mean(),
                                                'Cohen_Kappa_70%_Var': tempKFoldResults["Cohen_Kappa_70%"].var(),
                                                'All_Cohen_Kappa_60%': tempKFoldResults["Cohen_Kappa_60%"].tolist(),
                                                'Cohen_Kappa_60%': tempKFoldResults["Cohen_Kappa_60%"].mean(),
                                                'Cohen_Kappa_60%_Var': tempKFoldResults["Cohen_Kappa_60%"].var(),
                                                'All_Cohen_Kappa_50%': tempKFoldResults["Cohen_Kappa_50%"].tolist(),
                                                'Cohen_Kappa_50%': tempKFoldResults["Cohen_Kappa_50%"].mean(),
                                                'Cohen_Kappa_50%_Var': tempKFoldResults["Cohen_Kappa_50%"].var(),
                                                "AUC-ROC": tempKFoldResults["AUC-ROC"].mean(), 
                                                "AUC_Var": tempKFoldResults["AUC-ROC"].var(),
                                                "All_AUC": tempKFoldResults["AUC-ROC"].tolist(),
                                               'Precision_5%': tempKFoldResults["Precision_5%"].mean(),
                                               'Precision_10%': tempKFoldResults["Precision_10%"].mean(),
                                                'Precision_20%': tempKFoldResults["Precision_20%"].mean(),
                                                'Precision_30%': tempKFoldResults["Precision_30%"].mean(),
                                                'Precision_40%': tempKFoldResults["Precision_40%"].mean(),
                                               'Precision_50%': tempKFoldResults["Precision_50%"].mean(),
                                                'Precision_50%_Var': tempKFoldResults["Precision_50%"].var(),
                                                'All_Precision_50%': tempKFoldResults["Precision_50%"].tolist(),
                                               'Precision_40%': tempKFoldResults["Precision_40%"].mean(),
                                                'Precision_40%_Var': tempKFoldResults["Precision_40%"].var(),
                                                'All_Precision_40%': tempKFoldResults["Precision_40%"].tolist(),
                                                'Precision_30%': tempKFoldResults["Precision_30%"].mean(),
                                                'Precision_30%_Var': tempKFoldResults["Precision_30%"].var(),
                                                'All_Precision_30%': tempKFoldResults["Precision_30%"].tolist(),
                                                'Precision_60%': tempKFoldResults["Precision_60%"].mean(),
                                                'Precision_60%_Var': tempKFoldResults["Precision_60%"].var(),
                                                'All_Precision_60%': tempKFoldResults["Precision_60%"].tolist(),
                                                'Precision_70%': tempKFoldResults["Precision_70%"].mean(),
                                                'Precision_70%_Var': tempKFoldResults["Precision_70%"].var(),
                                                'All_Precision_70%': tempKFoldResults["Precision_70%"].tolist(),
                                                'Precision_80%': tempKFoldResults["Precision_80%"].mean(),
                                                'Precision_80%_Var': tempKFoldResults["Precision_80%"].var(),
                                                'All_Precision_80%': tempKFoldResults["Precision_80%"].tolist(),
                                                'Precision_90%': tempKFoldResults["Precision_90%"].mean(),
                                                'Precision_90%_Var': tempKFoldResults["Precision_90%"].var(),
                                                'All_Precision_90%': tempKFoldResults["Precision_90%"].tolist(),
                                                'Precision_95%': tempKFoldResults["Precision_95%"].mean(),
                                                'Precision_95%_Var': tempKFoldResults["Precision_95%"].var(),
                                                'All_Precision_95%': tempKFoldResults["Precision_95%"].tolist(),
                                               'Recall_5%': tempKFoldResults["Recall_5%"].mean(),
                                               'Recall_10%': tempKFoldResults["Recall_10%"].mean(),
                                                'Recall_20%': tempKFoldResults["Recall_20%"].mean(),
                                                'Recall_30%': tempKFoldResults["Recall_30%"].mean(),
                                                'Recall_40%': tempKFoldResults["Recall_40%"].mean(),
                                               'Recall_50%': tempKFoldResults["Recall_50%"].mean(),
                                               'Recall_50%_Var': tempKFoldResults["Recall_50%"].var(),
                                                'All_Recall_50%': tempKFoldResults["Recall_50%"].tolist(),
                                                'Recall_40%': tempKFoldResults["Recall_40%"].mean(),
                                               'Recall_40%_Var': tempKFoldResults["Recall_40%"].var(),
                                                'All_Recall_40%': tempKFoldResults["Recall_40%"].tolist(),
                                                'Recall_30%': tempKFoldResults["Recall_30%"].mean(),
                                               'Recall_30%_Var': tempKFoldResults["Recall_30%"].var(),
                                                'All_Recall_30%': tempKFoldResults["Recall_30%"].tolist(),
                                                'Recall_60%': tempKFoldResults["Recall_60%"].mean(),
                                               'Recall_60%_Var': tempKFoldResults["Recall_60%"].var(),
                                                'All_Recall_60%': tempKFoldResults["Recall_60%"].tolist(),
                                                'Recall_70%': tempKFoldResults["Recall_70%"].mean(),
                                               'Recall_70%_Var': tempKFoldResults["Recall_70%"].var(),
                                                'All_Recall_70%': tempKFoldResults["Recall_70%"].tolist(),
                                                'Recall_80%': tempKFoldResults["Recall_80%"].mean(),
                                                'Recall_80%_Var': tempKFoldResults["Recall_80%"].var(),
                                                'All_Recall_80%': tempKFoldResults["Recall_80%"].tolist(),
                                                'Recall_90%': tempKFoldResults["Recall_90%"].mean(),
                                               'Recall_90%_Var': tempKFoldResults["Recall_90%"].var(),
                                                'All_Recall_90%': tempKFoldResults["Recall_90%"].tolist(),
                                                'Recall_95%': tempKFoldResults["Recall_95%"].mean(),
                                               'Recall_95%_Var': tempKFoldResults["Recall_95%"].var(),
                                                'All_Recall_95%': tempKFoldResults["Recall_95%"].tolist()},
                                                ignore_index=True)
            

    return ModelResults





# Calls the function that will train and test all models
start = time.time()
ModelResults = pd.DataFrame()
ModelResults = RunThroughModelInputs(ModelResults, Features, Target, WhichDataBase)
print ('Took this many minutes to run: ', ((time.time() - start)/60) )



#Save the results of the runs
timestr = time.strftime("_%Y_%m_%d")
ModelResults.to_csv('/ihome/tgelsy/cjs180/Results_V14/ModelResults' + timestr + '.csv', index=False)

