import numpy as np
import pandas as pd

from dash.dependencies import Input, Output, State
from main.utils import MAX_LEN, PAGE_SIZE, color_cells, error_dict_to_arr, split_error_dict_per_metric


def get_cb_update_table(app):

    # ============================== UPDATE TABLE =============================================
    
    @app.callback(Output('display-table', 'data'),
                Output('display-table', 'style_data_conditional'),
                Output('display-table', 'page_count'),
                Input('display-table', 'page_current'),
                State('memory-error', 'data'),
                State('memory-dataset', 'data'),
                State('dirty-row-checkbox', 'value'))
    def update_table(page_current, errors, data, value):
        """ Updates displayed rows when switching pages, displays dirty cells with red background color """

        if page_current is None:
            page_current = 0

        df = pd.DataFrame(data)

        current_page_df = df.iloc[page_current * PAGE_SIZE:(page_current + 1) * PAGE_SIZE]
        style_data = None
        page_count = int(len(df) / PAGE_SIZE) if len(df) < MAX_LEN else MAX_LEN / PAGE_SIZE

        if errors is not None:
            errors = error_dict_to_arr(split_error_dict_per_metric(errors), df.shape)
            if not value:
                error_arr = np.array(errors)
            else:
                error_arr = np.array([arr for arr in errors if 1 in arr])
                index_arr = np.array([idx for idx, arr in enumerate(errors) if 1 in arr])

                df = df.iloc[index_arr, :]
                current_page_df = df.iloc[page_current * PAGE_SIZE:(page_current + 1) * PAGE_SIZE]
                page_count = int(len(df) / PAGE_SIZE) if len(df) < MAX_LEN else MAX_LEN / PAGE_SIZE

            color_array = error_arr[page_current * PAGE_SIZE:(page_current + 1) * PAGE_SIZE]
            style_data = color_cells(color_array, current_page_df, MAX_LEN)

        return current_page_df.to_dict('records'), style_data, page_count

    # =============================== DATATABLE FIRST PAGE ====================================
        
    @app.callback(Output('display-table', 'page_current'),
              Input('memory-error', 'data'),
              Input('dirty-row-checkbox', 'value'), prevent_initial_call=True)
    def datatable_first_page(data, value):
        """ if errors detected or checkbox (for only showing dirty rows) clicked,
        switch datatable to 1st page to display dirty cells """
        return 0
