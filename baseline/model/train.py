####################################################################################################################
# Train: Implement a model training method for regression, binary classification, and multi-class classification
# Authors: Mohamed Abdelaal
# Date: June 2022
# Software AG
# All Rights Reserved
####################################################################################################################

import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, mean_squared_error
from scikeras.wrappers import KerasClassifier, KerasRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from baseline.setup.utils import load_detections
from baseline.setup.detectors.detect_method import DATA_PATH, EXP_PATH
from baseline.dataset.preprocess import preprocess
from baseline.dataset.dataset import Dataset
from baseline.model.build_model import build_model
from baseline.model.hyperparams import get_hyperparams
from baseline.model.metrics import evaluate_model
from baseline.model.utils import create_results_path, store_results_csv, plot_learning_curves, ExperimentType, \
    ExperimentName

from main.apicalls import api_repair, api_basic_detection


# ============================== FIT MODEL ===============================================

def fit_model(X_train, y_train, X_valid, y_valid, ml_task, ml_model, epochs, hyperparams): 
    
    if ml_model == 'Neural Network': 
        # Create a wrapper around the Keras model
        if ml_task == 'regression':
            model = KerasRegressor(build_model, input_shape=X_train.shape[1:], learning_rate=hyperparams['learning_rate'],
                                n_hidden=hyperparams['n_hidden'], n_neurons=hyperparams['n_neurons'])

        elif ml_task in ['binary_classification', 'multiclass_classification']:
            nb_classes = 2 if ml_task == 'binary_classification' else y_train.shape[1]
            model = KerasClassifier(build_model, input_shape=X_train.shape[1:], ml_task=ml_task, nb_classes=nb_classes,
                                    learning_rate=hyperparams['learning_rate'], n_hidden=hyperparams['n_hidden'], n_neurons=hyperparams['n_neurons'])
        else:
            raise NotImplemented

        model.fit(X_train, y_train, epochs=epochs, validation_data=(X_valid, y_valid))
        return model
    #
    elif ml_model == 'Random Forest':

        if ml_task == 'regression':
            model = RandomForestRegressor(n_estimators=hyperparams['n_estimators'])
        elif ml_task in ['binary_classification', 'multiclass_classification']:
            model = RandomForestClassifier(n_estimators=hyperparams['n_estimators'])
        else:
            raise NotImplementedError
        
        model.fit(X_train, y_train)
        return model
    #
    elif ml_model == 'Decision Tree':
        
        if ml_task == 'regression':
            model = DecisionTreeRegressor()
        elif ml_task in ['binary_classification', 'multiclass_classification']:
            model = DecisionTreeClassifier()
        else:
            raise NotImplementedError
        
        model.fit(X_train, y_train)
        return model
    #
    else:
        raise NotImplementedError
    
    # ====================================== TRAIN MODEL ==========================================
    
def train_model(data_name,
                ml_model,
                tune_params=False,
                exp_name='',
                exp_type='',
                nb_trails=10,
                epochs=500,
                cv=5,
                verbose=True,
                detectors=[],
                repair_tools=[]):
    """
    Train a regressor/classifier using Keras
    """

    # Load default hyperparams or tune them
    best_cleaning_tools, hyperparams = get_hyperparams(data_name, ml_model, nb_trails=nb_trails, cv=cv, detectors=detectors, repair_tools=repair_tools, tune_params=tune_params,
                                  verbose=verbose)

    print("best cleaning tools: ", best_cleaning_tools)
    
    # Check if detections are already exist
    # Create detections path
    detections_path = Path(os.path.join(os.path.dirname(__file__), "..", "..", 'data', 'processed', data_name, best_cleaning_tools['detection_tool'], 'detections.csv'))
    
    if detections_path.exists():
        # Read the existing detections by the error detection tool
        error_dict = load_detections(detections_path)
    else:
        # Run the error detection tool
        error_dict, raha_sample_tuple, tuples_left = api_basic_detection(detection_methods=[best_cleaning_tools['detection_tool']])
    
    #error_dict, raha_sample_tuple, tuples_left = api_basic_detection(detection_methods=[best_cleaning_tools['detection_tool']])
    
    if error_dict is not None:
            
        repaired_path = api_repair(best_cleaning_tools['repair_tool'], detection_method=best_cleaning_tools['detection_tool'])
            
        if repaired_path is not None:
                
            repaired_df = pd.read_csv(repaired_path, header="infer", encoding="utf-8", keep_default_na=False, low_memory=False)

            # Initialize a dictionary for storing the various metrics
            metrics_dict = {}

            # Extract hyperparams
            learning_rate = hyperparams['learning_rate']
            n_hidden = hyperparams['n_hidden']
            n_neurons = hyperparams['n_neurons']

            # Create a Dataset object to retrieve the labels list
            data_obj = Dataset(data_name)
            labels = data_obj.cfg.labels
            ml_task = data_obj.cfg.ml_task

            # Prepare the data
            X_train_full, y_train_full, X_test, y_test = preprocess(repaired_df, labels_list=labels, ml_task=ml_task)
            # Extract a validation set
            X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)

            # Convert features to array if they are sparse matrices
            X_train = X_train if type(X_train) == np.ndarray else X_train.toarray()
            X_valid = X_valid if type(X_valid) == np.ndarray else X_valid.toarray()
            X_test = X_test if type(X_test) == np.ndarray else X_test.toarray()

            # Train the model
            start = time.time()
            model = fit_model(X_train, y_train, X_valid, y_valid, ml_task, ml_model, epochs, hyperparams)

            # Estimate the training time
            metrics_dict.update(training_time=time.time()-start)
            metrics_dict.update(epochs=epochs)
            metrics_dict.update(learning_rate=learning_rate)
            metrics_dict.update(n_hidden=n_hidden)
            metrics_dict.update(n_neurons=n_neurons)
            metrics_dict.update(timestamp=datetime.now())

            # Evaluate the models
            prediction = model.predict(X_test)
            # Evaluate the predictive accuracy
            metrics_dict = evaluate_model(metrics_dict, y_test, prediction, ml_task, verbose=True)

            # Prepare a path to store the results
            filename = exp_type + '_' + data_name + '.csv'
            results_path = create_results_path(data_name, exp_name, filename)

            # Store the results
            store_results_csv(metrics_dict, results_path)
            
            return best_cleaning_tools, metrics_dict, "Successul Training!"

        else:
            return {}, {}, "No repaired data!"  
    
    else:
        return {}, {}, "No detections found!"       


if __name__ == '__main__':
    # Get the data path
    dataset_name = 'housing'

    dataset_path = os.path.abspath(os.path.join(DATA_PATH, dataset_name))
    # Retrieve the dirty and clean data
    data_path = os.path.abspath(os.path.join(DATA_PATH, dataset_name, 'clean.csv'))

    # Load the data
    dataset_df = pd.read_csv(data_path, header="infer", encoding="utf-8", low_memory=False)

    # Train a model
    experiment_name = ExperimentName.MODELING.__str__()
    experiment_type = ExperimentType.GROUND_TRUTH.__str__()
    train_model(dataset_name,
                tune_params=False,
                exp_name=experiment_name,
                exp_type=experiment_type,
                verbose=True,
                epochs=10)

    # experiment_type: baseline, ground_truth, other baselines
