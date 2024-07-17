#####################################################################
# Hyperparams: Implement an Optuna-based hyperparameter tuning method
# Authors: Mohamed Abdelaal
# Date: June 2022
# Software AG
# All Rights Reserved
#####################################################################

import os
import json
import sys
import numpy as np
import pandas as pd
import optuna
from pathlib import Path
from baseline.dataset.dataset import Dataset
from baseline.dataset.preprocess import preprocess
from sklearn.model_selection import train_test_split, cross_val_score
from baseline.model.build_model import build_model
from scikeras.wrappers import KerasClassifier, KerasRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from baseline.setup.detectors.detect_method import DetectMethod, EXP_PATH, DATA_PATH
from baseline.setup.utils import load_detections
from main.apicalls import api_repair, api_basic_detection


def get_hyperparams(dataset_name, ml_model, nb_trails, cv=5, detectors=[], repair_tools=[], tune_params=False, verbose=True):
    """
    Load the default hyperparams or tune them. The action is based on the predicate tune_params
    """
    # Initialize a dict to store the hyperparams
    hyperparams = {}

    if tune_params:
        # Check if the tuned parameters already exist
        tuned_path = os.path.join(EXP_PATH, 'hyperparams', dataset_name, 'tuned.json')
        if os.path.exists(tuned_path):
            # Load the JSON file
            with open(tuned_path, 'r') as json_file:
                hyperparams = json.load(json_file)
                if hyperparams:
                    if verbose:
                        print("Hyperparameters already exist for the {} dataset".format(dataset_name))
                    return hyperparams
                else:
                    # Remove the JSON file if it is empty, before running the hyperparams tuning method
                    os.remove(tuned_path)
        else:
            if verbose:
                print("Optuna: Start hyperparameters tuning ..")
    else:
        # Load the default parameters if tuning is not required
        if verbose:
            print("Loading the default hyperparameters ..")
        default_path = os.path.join(EXP_PATH, 'hyperparams', 'default.json')
        with open(default_path, 'r') as json_file:
            return json.load(json_file)

    # Execute Optuna to tune the hyperparameters
    print("Finding the best error detection and repair combination ...")
    hyperparams = tune_hyperparams(dataset_name, ml_model, nb_trails=nb_trails, cv=cv, detectors=detectors, repair_tools=repair_tools)

    return hyperparams


def tune_hyperparams(dataset_name, ml_model, nb_trails=10, cv=5, detectors=[], repair_tools=[]):
    """
    Use Optuna to tune the hyperparameters of a model
    """
    
    # Create a Dataset object to retrieve the labels list
    print("Creating a Data object ..")
    print("Dataset Name: ", dataset_name)
    data_obj = Dataset(dataset_name)
    labels = data_obj.cfg.labels
    ml_task = data_obj.cfg.ml_task

    print("Labels:", labels)
    print("ml_task:", ml_model)
    
    # Load the default hyperparameters
    default_path = os.path.join(EXP_PATH, 'hyperparams', 'default.json')
    with open(default_path, 'r') as json_file:
        default_params = json.load(json_file)
        
    
    # =============================================== RANDOM FOREST =================================================    
       
    def objective_random_forest(trial):
        
        # Select a data cleaning method
        detection_tool = trial.suggest_categorical('detection_tool', detectors)
        repair_tool = trial.suggest_categorical('repair_tool', repair_tools)
        
        # Check if detections are already exist
        # Create detections path
        detections_path = Path(os.path.join(os.path.dirname(__file__), "..", "..", 'data', 'processed', dataset_name, detection_tool, 'detections.csv'))
        
        if detections_path.exists():
            # Read the existing detections by the error detection tool
            error_dict = load_detections(detections_path)
        else:
            # Run the error detection tool
            error_dict, raha_sample_tuple, tuples_left = api_basic_detection(detection_methods=[detection_tool])
        
        if error_dict is not None:
         
            repaired_path = api_repair(repair_tool, detection_method=detection_tool)
                
            if repaired_path is not None:
                    
                repaired_df = pd.read_csv(repaired_path, header="infer", encoding="utf-8", keep_default_na=False, low_memory=False)

                if not repaired_df.empty:
                    # Prepare the data
                    X_train_full, y_train_full, _, _ = preprocess(repaired_df, labels_list=labels, ml_task=ml_task)
        
                    # Hyperparameters to be tuned for Random Forest
                    #n_estimators = trial.suggest_int('n_estimators', 50, 1000)
                    #max_depth = trial.suggest_int('max_depth', 5, 50)
                    if ml_task == 'regression':
                        model = RandomForestRegressor(n_estimators=default_params['n_estimators'])
                        # Evaluate the model using cross-validation
                        rf_scores = cross_val_score(model, X_train_full, y_train_full, cv=cv, scoring='neg_mean_squared_error')
                    else:
                        nb_classes = 2 if ml_task == 'binary_classification' else y_train_full.shape[1]
                        model = RandomForestClassifier(n_estimators=default_params['n_estimators'])
                        rf_scores = cross_val_score(model, X_train_full, y_train_full, cv=cv, scoring='f1') if nb_classes == 2 else \
                                cross_val_score(model, X_train_full, y_train_full, cv=cv, scoring='f1_micro')
                    
                    return rf_scores.mean()
                else:
                    # if repaired_df empty
                    return 0
        else:
            # if no detections
            return 0

    # =============================================== Decision Tree ==================================================

    def objective_decision_tree(trial):
        
        # Select a data cleaning method
        detection_tool = trial.suggest_categorical('detection_tool', detectors)
        repair_tool = trial.suggest_categorical('repair_tool', repair_tools)
        
        # Check if detections are already exist
        # Create detections path
        detections_path = Path(os.path.join(os.path.dirname(__file__), "..", "..", 'data', 'processed', dataset_name, detection_tool, 'detections.csv'))
        
        if detections_path.exists():
            # Read the existing detections by the error detection tool
            error_dict = load_detections(detections_path)
        else:
            # Run the error detection tool
            error_dict, raha_sample_tuple, tuples_left = api_basic_detection(detection_methods=[detection_tool])
        
        if error_dict is not None:
         
            repaired_path = api_repair(repair_tool, detection_method=detection_tool)
                
            if repaired_path is not None:
                    
                repaired_df = pd.read_csv(repaired_path, header="infer", encoding="utf-8", keep_default_na=False, low_memory=False)

                if not repaired_df.empty:
                    # Prepare the data
                    X_train_full, y_train_full, _, _ = preprocess(repaired_df, labels_list=labels, ml_task=ml_task)
                    
                    # Hyperparameters to be tuned for Decision Tree
                    #max_depth = trial.suggest_int('max_depth', 1, 32)
                    if ml_task == 'regression':
                        model = DecisionTreeRegressor()
                        # Evaluate the model using cross-validation
                        dt_scores = cross_val_score(model, X_train_full, y_train_full, cv=cv, scoring='neg_mean_squared_error')
                    else:
                        nb_classes = 2 if ml_task == 'binary_classification' else y_train_full.shape[1]
                        model = DecisionTreeClassifier()
                        dt_scores = cross_val_score(model, X_train_full, y_train_full, cv=cv, scoring='f1') if nb_classes == 2 else \
                                cross_val_score(model, X_train_full, y_train_full, cv=cv, scoring='f1_micro')
        
                    return dt_scores.mean()
                else:
                    # if repaired_df empty
                    return 0
        else:
            # if no detections
            return 0

    # =========================================== Neural Network ======================================================
        
    def objective(trial):
    
        # Select a data cleaning method
        detection_tool = trial.suggest_categorical('detection_tool', detectors)
        repair_tool = trial.suggest_categorical('repair_tool', repair_tools)
        
        # Check if detections are already exist
        # Create detections path
        detections_path = Path(os.path.join(os.path.dirname(__file__), "..", "..", 'data', 'processed', dataset_name, detection_tool, 'detections.csv'))
        
        if detections_path.exists():
            # Read the existing detections by the error detection tool
            error_dict = load_detections(detections_path)
        else:
            # Run the error detection tool
            error_dict, raha_sample_tuple, tuples_left = api_basic_detection(detection_methods=[detection_tool], uiltags=[])
        
        if error_dict is not None:
         
            repaired_path = api_repair(repair_tool, detection_method=detection_tool)
                
            if repaired_path is not None:
                    
                repaired_df = pd.read_csv(repaired_path, header="infer", encoding="utf-8", keep_default_na=False, low_memory=False)

                if not repaired_df.empty:
                    # Prepare the data
                    X_train_full, y_train_full, _, _ = preprocess(repaired_df, labels_list=labels, ml_task=ml_task)

                    # Create a wrapper around the Keras model
                    if ml_task == 'regression':
                        model = KerasRegressor(build_model, input_shape=X_train_full.shape[1:], learning_rate=default_params[
                            'learning_rate'], n_hidden=default_params['n_hidden'], n_neurons=default_params['n_neurons'])
                        # Evaluate the model using cross-validation
                        scores = cross_val_score(model, X_train_full, y_train_full, cv=cv, scoring='neg_mean_squared_error')

                    elif ml_task in ['binary_classification', 'multiclass_classification']:
                        nb_classes = 2 if ml_task == 'binary_classification' else y_train_full.shape[1]
                        model = KerasClassifier(build_model, input_shape=X_train_full.shape[1:], ml_task=ml_task, nb_classes=nb_classes,
                                                learning_rate=default_params['learning_rate'], n_hidden=default_params['n_hidden'],
                                                n_neurons=default_params['n_neurons'])
                        # Evaluate the model using cross-validation
                        scores = cross_val_score(model, X_train_full, y_train_full, cv=cv, scoring='f1') if nb_classes == 2 else \
                                cross_val_score(model, X_train_full, y_train_full, cv=cv, scoring='f1_micro')
                    else:
                        raise NotImplemented

                    return scores.mean()
                else:
                    # if repaired_df empty
                    return 0
        else:
            # if no detections
            return 0
   
   # ============================================ OPTUNA OPTIMIZATION =================================================
    
    # Create a study object and specify the direction is 'maximize' since we want to maximize accuracy
    study = optuna.create_study(direction='minimize') if ml_task == 'regression' else \
            optuna.create_study(direction='maximize')
    
    if ml_model == 'Neural Network':
        # Run the optimization
        study.optimize(objective, n_trials=nb_trails)
    elif ml_model == 'Random Forest':
        # Run the optimization
        study.optimize(objective_random_forest, n_trials=nb_trails)
    elif ml_model == 'Decision Tree':
        # Run the optimization
        study.optimize(objective_decision_tree, n_trials=nb_trails)
    else:
        raise NotImplementedError
    
    # Get the best parameters
    best_cleaning_method = study.best_params

    return best_cleaning_method, default_params


if __name__ == '__main__':
    # Get the data path
    dataset_name = 'nasa'
    dataset_path = os.path.abspath(os.path.join(DATA_PATH, dataset_name))
    # Retrieve the dirty and clean data
    data_path = os.path.abspath(os.path.join(DATA_PATH, dataset_name, 'clean.csv'))
    # Load the data
    data_df = pd.read_csv(data_path, header="infer", encoding="utf-8", low_memory=False)
    print(EXP_PATH)
    # Train a model
    get_hyperparams(data_df, dataset_name, nb_trails=50, epochs=100, tune_params=True, verbose=False)

