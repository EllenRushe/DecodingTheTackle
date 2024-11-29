from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from imblearn.over_sampling import SMOTE,  RandomOverSampler
from imblearn.under_sampling import  RandomUnderSampler, ClusterCentroids
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn import tree
import pandas as pd
import numpy as np
import dtreeviz
import random
import pickle
import json
import os
import re

from utils import get_feature_number, get_targets, get_features


def grid_search(X_train, y_train, prefix, class_names, feature_names):

    '''
    params:
        X_train (NumPy matrix): Training feature matrix in one-hot encoded format.
        y_train (NumPy array): Target feature vector. 
        prefix (str): String to identify model. 
        class_names (list): Human-readable names for each class level in target feature. 
        feature_names (list): Human-readable names for each feature in feature matrix. 
    '''

    log_dir = "logs"
    if not os.path.exists(log_dir): os.mkdir(log_dir)
    nb_parameters = {"classification__fit_prior": (True, False)}
    lr_parameters = {
        'classification__max_iter': (100, 200, 300), 
        'classification__penalty':('l1', 'l2', 'elasticnet', None),
        'classification__solver': (
            'lbfgs', 
            'liblinear', 
            'newton-cg', 
            'newton-cholesky', 
            'sag', 
            'saga'
        )
    }

    dt_parameters = {
        'classification__criterion': ("gini","entropy","log_loss"),
        'classification__max_depth':[2, 3, 4, 5, 6],
        'classification__class_weight': ("balanced", None)
    }


    params = {
        'nb':nb_parameters,
        'lr':lr_parameters,
        'dt':dt_parameters,
    }
    clfs = {
        'nb':BernoulliNB(),
        'lr':LogisticRegression(),
        "dt":DecisionTreeClassifier(),
    }

    sampler_dict = {
        "random_under_sampling": RandomUnderSampler,
        "random_over_sampling": RandomOverSampler   
    }
    for sampler_k, sampler_obj in sampler_dict.items(): 
        # Dictionary to store the best parameters and best score for each model. 
        gs_results = {}
        # Initialise the best score overall for all classifiers to zero. 
        best_score_overall = 0
        best_model = ""
        for clf_name in sorted(clfs.keys()):
            model = Pipeline([
                ('sampling', sampler_obj(random_state=0)),
                ('classification', clfs[clf_name])
            ])

            # Initialise grid search using 5-fold cross validation for each classifier. 
            gs_cv = GridSearchCV(
                model, 
                param_grid=params[clf_name],
                cv=5,
                scoring='f1_macro'
            )
            # Run grid search. 
            gs_cv.fit(X_train, y_train)
            # Create dataframe with detailed grid-search results, for each classifier and identify with prefix. 
            cv_meta_data_df = pd.DataFrame.from_dict(gs_cv.cv_results_)
            cv_meta_data_df.to_csv(os.path.join(log_dir, "target_{}_{}_{}_grid_search_cross_val.csv".format(prefix, clf_name, sampler_k)))
            # Collect best parameters for this classifier along with the corresponding score for those parameters. 
            best_params = dict()
            for k, v in gs_cv.best_params_.items():
                best_params[k.split('__')[-1]] = v
            gs_results[clf_name] =  {
                'best_params':best_params,
                'best_score': gs_cv.best_score_,
                "sampler": sampler_k
            }
            # Check whether the best score obtained for this classifier is the highest so far. 
            if gs_cv.best_score_ > best_score_overall:
                #  If so, update the best score overall and the best classifier overall.
                best_score_overall = gs_cv.best_score_
                best_model = clf_name
        # Store to log. 
        gs_results["best_score_overall"] = best_score_overall
        gs_results["best_model"] = best_model

        # Write log to JSON. 
        with open(os.path.join(log_dir, "{}_{}_grid_search_results.json".format(prefix, sampler_k)), "w") as f:
            json.dump(gs_results, f)   
        
        # Read pre-generated random seeds for the purposes of reproducability.
        random_seeds = pd.read_csv("random_seeds.csv")["seeds"].values
        # Naive Bayes will not change based on different random seeds so it is excluded from this process. 
        # Decision trees can exhibit randomness in the case of ties which is likely when independent features are fully categorical and binary encoded.
        if gs_results["best_model"] == "nb":
            # Get the class reference to the best model from the clf dictionary and run with the best parameters of the best model. 
            clf = clfs[gs_results["best_model"]].__class__(**gs_results[gs_results["best_model"]]["best_params"])
            sm = sampler_obj(random_state=0)
            X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
            clf.fit(X_train_res, y_train_res)
            # Save model as pickle. 
            pickle.dump(clf,  open(os.path.join(model_dir, "{}_{}_{}.pickle".format(prefix, gs_results["best_model"], sampler_k)), 'wb'))
        else:
            # Iterate over random seeds. 
            for seed in random_seeds:
                # Pass random state. 
                clf = clfs[gs_results["best_model"]].__class__(random_state=seed, **gs_results[gs_results["best_model"]]["best_params"])
                sm = sampler_obj(random_state=0)
                X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
                clf.fit(X_train_res, y_train_res)
                pickle.dump(clf,  open(os.path.join(model_dir, "{}_{}_seed_{}_{}.pickle".format(prefix, gs_results["best_model"], seed, sampler_k)), 'wb'))

# Create log and meta-data directories. 
model_dir = "models"
image_dir = "images"
data_dir = "data"

if not os.path.exists(model_dir) : os.mkdir(model_dir)
if not os.path.exists(image_dir) : os.mkdir(image_dir)

# Human readable labels are used to aid in visualisations. 
with open(os.path.join(data_dir, "human_readable_labels_features.json"), 'rb') as f:
    readable_features_map = json.load(f)
    
train_df = pd.read_csv(os.path.join(data_dir, "train_31_vars.csv"))
test_df = pd.read_csv(os.path.join(data_dir, "test_31_vars.csv"))

# A subset of "feature of interest" is tested us used for some models. 
# Feature subset includes 0-30 inclusive. 
X_subset_features_28 = [str(i) for i in range(31) if i !=28]
X_subset_features_29 = [str(i) for i in range(31) if i !=29]
# Feature subset includes 0-30 excluding feature 30.
X_subset_features_30 = [str(i) for i in range(30)]

# Extract different training sets. 
X_train_subset_28, X_train_subset_28_df, X_train_subset_feature_28_names = get_features(train_df, X_subset_features_28)
X_train_subset_29, X_train_subset_29_df, X_train_subset_feature_29_names = get_features(train_df, X_subset_features_29)
X_train_subset_30, X_train_subset_30_df, X_train_subset_30_feature_names = get_features(train_df, X_subset_features_30)


X_train_subset_28_df_renamed = X_train_subset_28_df.rename(columns=readable_features_map)
X_train_subset_29_df_renamed = X_train_subset_29_df.rename(columns=readable_features_map)
X_train_subset_30_df_renamed = X_train_subset_30_df.rename(columns=readable_features_map)


# Different target feature are returned as a NumPy array, a DataFrame, along with the human-readable class levels. 
y_train_28, y_train_28_df, class_names_28= get_targets(
    train_df, 
    readable_features_map,
    target='28'
)


y_train_29, y_train_29_df, class_names_29= get_targets(
    train_df, 
    readable_features_map,
    target='29'
)
                        
y_train_30, y_train_30_df, class_names_30= get_targets(
    train_df, 
    readable_features_map,
    target='30'
) 


# # Calls 5-fold cross-validated grid search for each set independent training features and targets. 
grid_search(
    X_train_subset_28, 
    y_train_28, 
    "target_28", 
    class_names=class_names_28,
    feature_names=X_train_subset_28_df_renamed.columns
)


grid_search(
    X_train_subset_29, 
    y_train_29, 
    "target_29", 
    class_names=class_names_29,
    feature_names=X_train_subset_29_df_renamed.columns

)

grid_search(
    X_train_subset_30, 
    y_train_30, 
    "target_30", 
    class_names=class_names_30,
    feature_names=X_train_subset_30_df_renamed.columns

)
