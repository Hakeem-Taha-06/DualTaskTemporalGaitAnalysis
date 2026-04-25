"""
gait — Core pipeline modules for dual-task temporal gait analysis.

Modules:
    input_loader          Load raw coordinate data from Sports2D .trc files
    preprocessor          Optional smoothing / coordinate convention check
    event_detector        Detect heel-strike (HS) and toe-off (TO) events
    parameter_calculator  Compute stride-by-stride spatio-temporal parameters
    outlier_remover       Remove invalid strides (boundary, turning, etc.)
    aggregator            Aggregate stride-level data to participant summaries
    dtc_calculator        Compute Dual-Task Cost (DTC)
"""

from gait.input_loader import load_trc, load_from_dataframe
from gait.preprocessor import preprocess
from gait.event_detector import detect_events, events_to_gait_event_dict
from gait.parameter_calculator import calculate_parameters
from gait.outlier_remover import remove_outliers
from gait.aggregator import aggregate
from gait.dtc_calculator import calculate_dtc, dtc_summary_table
