# Standard
import sys
import pathlib
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add project root directory to sys.path
root_dir = str(pathlib.Path(__file__).parents[1])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
# Third-party
import numpy as np
# Local
from projects.gnn_material_patch.material_patch.patch_generator import \
    rotation_tensor_from_euler_angles
# =============================================================================
# Summary: Test tensor rotations
# =============================================================================
print(f'\nTENSOR ROTATIONS')
print(f'----------------')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set first-order tensor (basis a)
x_a = np.array((1, 0, 0))
# Set rotation Euler angles
euler_deg = np.array((0, 90, -90))
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Display
print(f'> Input data:')
print(f'\n  x_a = {x_a}')
print(f'\n  euler_deg = {euler_deg}')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set rotation tensor
rotation = rotation_tensor_from_euler_angles(euler_deg)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Rotate first-order tensor (basis a, active transformation)
xr_a = np.matmul(rotation, x_a)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Display
print(f'\n> Output data:')
print(f'\n  xr_a = {xr_a}')
print()