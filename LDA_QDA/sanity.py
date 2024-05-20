import sys, os
if sys.version_info[0] < 3:
    raise Exception("Python 3 not detected.")

import numpy as np
import matplotlib.pyplot as plt 
import sklearn
from scipy import io

CURR_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(CURR_DIR, 'data')

EXECTED_SHAPES = {
    'mnist' : {
        'training_data' : (60000, 784),
        'training_labels' : (60000, 1),
        'test_data' : (10000, 784)
    },
    'spam' : {
        'training_data' : (5172, 32),
        'training_labels' : (5172, 1),
        'test_data' : (5857, 32)
    }
}

for data_name in ["mnist", "spam"]:
    data = io.loadmat(os.path.join(DATA_DIR, "{}_data.mat".format(data_name)))
    print("loaded %s data!" % data_name)
    fields = "test_data", "training_data", "training_labels"
    for field in fields:
        assert field in data, "Missing {} field in {}!".format(field, data_name)
        expected_shape = EXECTED_SHAPES[data_name][field]
        actual_shape = data[field].shape
        assert EXECTED_SHAPES[data_name][field] == data[field].shape, "{} field of {} is wrong shape!\Expected {} but got {}".format(field, data_name, expected_shape, actual_shape)
print("All Good!")