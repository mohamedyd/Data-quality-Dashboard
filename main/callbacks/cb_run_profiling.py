from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State

from main.apicalls import api_data_profile


def get_cb_run_profiling(app):
    
    # ========================= DATA PROFILE ======================================
    
    @app.callback(Output('data-profile', 'src'),
              Output('alert-message', 'data', allow_duplicate=True),
              Input('memory-dataset', 'data'),
              State('memory-filename', 'data'), prevent_initial_call=True)
    def data_profile(data, dataset_name):
        """ Create profile (report containing information about dataset) of uploaded data using ydata_profiling """

        if data is None:
            raise PreventUpdate

        profile_path = api_data_profile()

        if profile_path is not None:
            return profile_path, None
        
        return None, "Error getting Data Profile."
