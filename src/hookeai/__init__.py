"""HookeAI: An open-source ADiMU framework"""
#
#                                                                       Modules
# =============================================================================
import sys
import pathlib
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add project root directory to sys.path
root_dir = str(pathlib.Path(__file__).parents[0])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from src.vegapunk import main
from src.vegapunk import data_generation
from src.vegapunk import ioput
from src.vegapunk import material_model_finder
from src.vegapunk import miscellaneous
from src.vegapunk import model_architectures
from src.vegapunk import simulators
from src.vegapunk import time_series_data
from src.vegapunk import user_scripts
from src.vegapunk import utilities
