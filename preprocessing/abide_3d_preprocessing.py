"""
==============================================================================
ABIDE-I 3D Multi-Planar Preprocessing Launcher
------------------------------------------------------------------------------
Author: Clint Loyed
Target Sites: NYU, UM_1, USM (~400 Subjects Total)
Target Resolution: Full HD (50, 224, 224, 1) 3D Volumetric Tensors

Default Action: Executes the 224x224 Full HD 3-University Preprocessing Pipeline!
==============================================================================
"""

import os
import sys

# Import and execute the 224x224 Full HD Multi-Site pipeline
from preprocessing.abide_3d_preprocessing_224_multisite import *
