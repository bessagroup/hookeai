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
from src.hookeai import data_generation
from src.hookeai import ioput
from src.hookeai import material_model_finder
from src.hookeai import miscellaneous
from src.hookeai import model_architectures
from src.hookeai import simulators
from src.hookeai import time_series_data
from src.hookeai import utilities
from src.hookeai import user_scripts
