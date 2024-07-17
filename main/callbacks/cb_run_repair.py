import os

from dash import html
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
from main.utils import combine_csv_files
from main.apicalls import api_repair

def get_cb_run_repair(app):

    # ============================== RUN ERROR REPAIR =====================================
    
    @app.callback(Output('repair-message', 'children'),
                Input('repair-button', 'n_clicks'),
                State('memory-dataset', 'data'),
                State('memory-filename', 'data'),
                State('memory-filepath', 'data'),
                State('repair-methods', 'value'),
                State('detection-method', 'value'), prevent_initial_call=True)
    def run_error_repair(n_clicks, dataset, filename, filepath, method, detection_method):
        if dataset is None:
            raise PreventUpdate

        if detection_method is None:
            return html.P("Please run error detection first")
        
        #detections_path = os.path.join(os.path.dirname(filepath), 'detections.csv')
        #if not os.path.exists(detections_path):
        #    html.P("Please run error detection first")
        
        if len(detection_method) == 1:

            repaired_path = api_repair(repair_method=method,detection_method=detection_method[0])

        else:
            # read detections from each csv file in each folder in processed/dataset_name
            # append all detections from all folders in one csv file with removing duplicates
            # save all the obtained detections in a csv file in the combined_dir, which is created to store the combined detections
            # repair detections in concatenated dir
            combine_csv_files(dataset_name=filename, combined_dir_name='combined_detections')
            repaired_path = api_repair(repair_method=method,detection_method='combined_detections')
            
        if repaired_path is not None:
            return html.P(f"Repair successful, saved results at {repaired_path}")
        else:
            return html.P("Repair unsuccessful")