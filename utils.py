import pandas as pd
import numpy as np
import re


def get_feature_number(feature_name):
    '''
    Extract the first integer number to appear in the string passed. 
    params: 
        feature_name (str): Feature with integer we want to extract. 

    returns:
        Returns str of first integer number (no matter how large). Note: 
        subsequent integer values after the first instance of a non-integer value 
        following the first integer will not be returned.
        or 
        None if no integer is present. 

    '''
    all_num_strs = re.findall(r"(\d+)",feature_name)
    if len(all_num_strs) > 0:
        return all_num_strs[0]
    else:
        None

def get_features(df, X_subset_features=[], target_feature='43.Tackle result'):
    '''
    Extract feature columns from DataFrame (i.e. remove target feature). 
    Will return all columns but the target feature column if no feature
    subset is passed. 
    params:
        df (DataFrame): DataFrame to extract feature columns from.
        X_subset_feature (list): List of numbers within the column names that are in feature set. 
        target_feature (str): Integer number in the feature name string (assumes that there 
                              is a number in the target name!!). 

    returns:
        X (NumPY): Feature matrix corresponding to filtered DataFrame. 
        X_df (DataFrame): Filtered DataFrame. 
        X_feature_names (list): List of full feature names for the feature subset. 
    '''
    X_feature_names = []
    if len(X_subset_features)==0:
        for feature_name in df.columns:
            target_feature_number = get_feature_number(target_feature)
            feature_number = get_feature_number(feature_name)
            if feature_number != target_feature_number:
                X_feature_names.append(feature_name)
    else:
        for feature_name in df.columns:
            for num in X_subset_features:
                feature_number = get_feature_number(feature_name)
                if num == feature_number:
                    X_feature_names.append(feature_name)
    X_df = df[X_feature_names]
    X = X_df.values
    return X, X_df, X_feature_names

def get_targets (df, readable_map, target):
    '''
    params:
        df (DataFrame): Get columns corresponding to target and format as a single column. Columns names have to 
                        be in "to_dummies" format, separated by '_' (e.g. feature1_0, feature_1_1, ..., feature_1_n).
        readable_map (dict): Dictionary from feature value to human readable feature values 
                             for all levels of target feature. 
        target (str): Number in target variable as a string, e.g. '43'.

    returns:
        y (NumPy array): Target feature as single column NumPy array. 
        y_df (DataFrame):  Target feature as single column DataFrame.
        y_names (list): List of human readable names for each level op the target feature in the order of the numericly 
                        encoded values. 
    '''
    # This will not work if the target is a single digit number or > two digits! 
    y_names_df =[x for x in df.columns if target in x]
    y_df = df[y_names_df]
    # If target columns are one-hot-encoded. 
    if len(y_df.columns) >1:
        y_names = [readable_map[key] for key in y_df.columns]
        y_df = pd.from_dummies(y_df , sep='_')
        
    else:
        # np.unique will return the unique values in order which means y_names is in order.
        y_names = [readable_map[y_df.columns[0]][i] for i in np.unique(y_df)]
    y = np.squeeze(y_df.values.astype(int))
    return y, y_df, y_names