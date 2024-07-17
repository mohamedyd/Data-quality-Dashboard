import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dash import dash_table, dcc
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State

from main.apicalls import api_basic_detection
from main.utils import error_dict_to_arr, split_error_dict_per_metric


def get_cb_run_detectors(app):
   
   # ============================= DETECT ERRORS ====================================
    
    @app.callback(Output('memory-error', 'data', allow_duplicate=True),
              Output('raha-signal', 'data'),
              Output('raha-tuple', 'data'),
              Output('tuples-left', 'data'),
              Output('alert-message', 'data', allow_duplicate=True),
              Input('detect-button', 'n_clicks'),
              State('memory-dataset', 'data'),
              State('memory-filename', 'data'),
              State('memory-filepath', 'data'),
              State('detection-method', 'value'),
              State('labeling-budget', 'value'),
              State('memory-uiltags', 'data'), prevent_initial_call=True)
    def detect_errors(n_clicks, dataset, filename, filepath, detection_method, labeling_budget, uiltags):
        """
        Runs various error detection methods as specified by detection_method array
        """

        # error_dict contains detection methods as keys and lists of dirty cell positions as values
        if detection_method is None:
            return None, None, None, None, "Enter at least one error detection method!"   
        
        if "Raha" in detection_method:
            # If 'raha' is selected but no budget is provided, prompt the user.
            if labeling_budget is None:
                return (None, None, None, None, "Please enter a labeling budget for Raha.")
        else:
            labeling_budget = 0
        
        error_dict, raha_sample_tuple, tuples_left = api_basic_detection(detection_method, uiltags, labeling_budget)

        if error_dict is not None:
            if tuples_left != 0:
                return error_dict, True, raha_sample_tuple, tuples_left, None
            else:
                return error_dict, None, None, None, None
        return None, None, None, None, "No Errors were found!"     

    # ============================ DISPLAY ERROR DETECTION RESULT =======================

    @app.callback(Output('error-detection-result', 'children'),
                Input('memory-error', 'data'),
                State('memory-dataset', 'data'), prevent_initial_call=True)
    def display_error_detection_result(errors, dataset):
        """ Displays error detection results on the dashboard """
        if errors is None:
            return None

        df = pd.DataFrame(dataset)
        error_arr = error_dict_to_arr(split_error_dict_per_metric(errors), df.shape)

        data = {'Column': df.columns, 'Error rate': np.round(np.mean(error_arr, axis=0), 3)}
        error_df = pd.DataFrame(data=data)

        return dash_table.DataTable(
            error_df.to_dict('records'),
            [{'name': i, 'id': i} for i in error_df.columns],
            style_table={'overflowX': 'auto'},
        )

    # ============================= ERROR DETECTION PLOTS ==================================
    
    
    @app.callback(Output('bar', 'children'),
                Output('donut', 'children'),
                Input('memory-error', 'data'),
                Input('graph-orientation', 'value'),
                State('memory-dataset', 'data'), prevent_initial_call=True)
    def error_detection_plots(errors, orientation, data):
        """ Display error detection results using a bar chart in the "Error Detection Results" tab and a donut chart
                on main page
        """
        if errors is None or data is None:
            return None, None

        df = pd.DataFrame(data)
        data_for_bar_chart = []

        errors = split_error_dict_per_metric(errors)

        for k, v in errors.items():
            # for error metric processed in current iteration, compute proportions of dirty cells in each column and
            #   append to data_for_bar_chart
            arr = error_dict_to_arr({k: v}, df.shape)
            error_proportion = np.mean(arr, axis=0)

            # array with rows = error metric and columns = columns of dataset
            data_for_bar_chart.append(list(error_proportion))

        index = pd.MultiIndex.from_product([list(errors.keys()), df.columns], names=["Error metric", "Column"])
        bar_chart_df = pd.DataFrame(index=index).reset_index()
        bar_chart_df["Error rate"] = np.array(data_for_bar_chart).flatten()

        # for each column in the dataset, make a bar chart to show which kinds of errors are present and in which proportion
        if orientation == 'Vertical':
            bar = px.bar(bar_chart_df, x="Column", y="Error rate", color="Error metric", barmode='group')
        else:
            bar = px.bar(bar_chart_df, x="Error rate", y="Column", color="Error metric", barmode='group', orientation='h')

        # donut chart
        error_array_all_metrics = error_dict_to_arr(errors, df.shape)  # array with 1s in positions of dirty cells
        error_rate_whole_dataset = np.sum(error_array_all_metrics)/error_array_all_metrics.size
        donut = go.Figure(data=[go.Pie(labels=["Dirty cells", "Clean cells"],
                                    values=[error_rate_whole_dataset, 1-error_rate_whole_dataset], hole=.3)])
        donut.update_layout(margin=dict(l=5, r=5, t=0, b=0), legend_y=0.8)

        return dcc.Graph(figure=bar), dcc.Graph(figure=donut, style={'height': '30vh'})

