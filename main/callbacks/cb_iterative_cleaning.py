import pandas as pd
from main.utils import ml_models, error_detection, error_repair
from main.apicalls import api_basic_detection, api_repair
from dash import html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from baseline.model.train import train_model
from baseline.model.utils import ExperimentType, ExperimentName

def get_cb_iterative_cleaning(app):
   
    @app.callback(Output('memory-error', 'data', allow_duplicate=True),
              Output('raha-signal', 'data', allow_duplicate=True),
              Output('raha-tuple', 'data', allow_duplicate=True),
              Output('tuples-left', 'data', allow_duplicate=True),
              Output('ic-message', 'children'),
              Input('iterative-clean-button', 'n_clicks'),
              State('ml-model-dropdown', 'value'),           # Get the value of the dropdown when the button is clicked
              State('memory-dataset', 'data'),
              State('memory-filename', 'data'),
              State('memory-filepath', 'data'),
              State('detection-method', 'value'),
              State('labeling-budget', 'value'), prevent_initial_call=True)
    def initialize_ic(n_clicks, ml_model_value, dataset, filename, filepath, detection_method, labeling_budget):
        """
        Runs various error detection methods as specified by detection_method array
        """

        detection_method = ['Raha']
        
        if ml_model_value is None:
           # If no model is selected, inform the user to select a model
           return (None, None, None, None, "Please select a machine learning model to continue.")
        
        if labeling_budget is None:
            return (None, None, None, None, "Please enter a labeling budget for Raha.")

        error_dict, raha_sample_tuple, tuples_left = api_basic_detection(detection_method, labeling_budget=labeling_budget)

        if error_dict is not None:
            if tuples_left != 0:
                return (error_dict, True, raha_sample_tuple, tuples_left, None)
            else:
                return (error_dict, None, None, None, None)
        return (None, None, None, None, "No Errors were found!")     
    
    
    # ============================= RUN INTERATIVE CLEANING ===============================================


    @app.callback(
    Output('ic-message', 'children', allow_duplicate=True),  # Update the children property of the message div
    Input('ic-signal', 'data'),
    State('num-training-iterations', 'value'),
    State('num-optuna-trials', 'value'),
    State('num-optuna-cv', 'value'),
    State('iterative-clean-button', 'n_clicks'),
    State('ml-model-dropdown', 'value'),           # Get the value of the dropdown when the button is clicked
    State('memory-dataset', 'data'),
    State('memory-filename', 'data'),
    State('memory-filepath', 'data'),
    prevent_initial_call=True                      # Prevent the callback from being called at startup
    )
    def run_iterative_cleaning(ic_signal, n_epochs, n_trials, cv, n_clicks, ml_model_value, dataset, filename, filepath):
        print("IC Signal: ", ic_signal)
        print("ML Model: ", ml_model_value)
        if ml_model_value is None or n_clicks == 0:
            raise PreventUpdate
        else:
            detection_tools_list = list(error_detection.keys())
            repair_tools_list = list(error_repair.keys())

            best_cleaning_tools, eval_metrics, msg = train_model(filename,
                                                                ml_model=ml_model_value, 
                                                                tune_params=True, 
                                                                exp_name=ExperimentName.MODELING.__str__(), 
                                                                exp_type=ExperimentType.GROUND_TRUTH.__str__(), 
                                                                verbose=True, epochs=n_epochs, nb_trails=n_trials, cv=cv,
                                                                detectors=detection_tools_list,
                                                                repair_tools=repair_tools_list)
                    
            return f"Repair successful (best cleaning tools:{best_cleaning_tools}), {msg}: {eval_metrics}"





    # @app.callback(
    # Output('iterative-clean-message', 'children'),  # Update the children property of the message div
    # Input('iterative-clean-button', 'n_clicks'),   # Listen for clicks on the button
    # State('ml-model-dropdown', 'value'),           # Get the value of the dropdown when the button is clicked
    # State('memory-dataset', 'data'),
    # State('memory-filename', 'data'),
    # State('memory-filepath', 'data'),
    # prevent_initial_call=True                      # Prevent the callback from being called at startup
    # )
    # def run_iterative_cleaning(n_clicks, ml_model_value, dataset, filename, filepath):
    #     if ml_model_value is None:
    #         # If no model is selected, inform the user to select a model
    #         return "Please select a machine learning model to continue."
    #     else:
    #         detection_tools_list = list(error_detection.keys())
    #         repair_tools_list = list(error_repair.keys())

    #         best_cleaning_tools, eval_metrics, msg = train_model(filename, 
    #                                                             tune_params=True, 
    #                                                             exp_name=ExperimentName.MODELING.__str__(), 
    #                                                             exp_type=ExperimentType.GROUND_TRUTH.__str__(), 
    #                                                             verbose=True, epochs=10,
    #                                                             detectors=detection_tools_list,
    #                                                             repair_tools=repair_tools_list)
                    
    #         return f"Repair successful (best cleaning tools:{best_cleaning_tools}), {msg}: {eval_metrics}"