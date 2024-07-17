from .cb_upload_dataset import get_cb_upload_dataset
from .cb_run_repair import get_cb_run_repair
from .cb_update_table import get_cb_update_table
from .cb_fd_detection import get_cb_fd_detection
from .cb_datasheets import get_cb_datasheets
from .cb_run_raha import get_cb_raha
from .cb_run_detectors import get_cb_run_detectors
from .cb_run_profiling import get_cb_run_profiling
from .cb_alerts import get_cb_alerts
from .cb_iterative_cleaning import get_cb_iterative_cleaning
from .cb_upload_state_management import get_cb_upload_state_management

callbacks = [
    get_cb_upload_dataset,
    get_cb_fd_detection,
    get_cb_update_table,
    get_cb_run_repair,
    get_cb_datasheets,
    get_cb_raha,
    get_cb_run_detectors,
    get_cb_run_profiling,
    get_cb_alerts,
    get_cb_iterative_cleaning,
    get_cb_upload_state_management
]