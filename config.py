import numpy as np

NUM_CLASSES = 20
OUTPUT_TENSOR_NAMES = ['mixed_2/join:0', 'mixed_7/join:0', 'mixed_10/join:0']
APPEND_POOL = [2,2]
A_R = [1.0, 2.0, 3.0, 1./2., 1./3.]
BBOX_RATIOS = [(np.sqrt(a), 1/np.sqrt(a)) for a in A_R]
NUM_BOXES = len(BBOX_RATIOS)
OUTPUT_DIMS = NUM_CLASSES + 5*NUM_BOXES
