import numpy as np
from pathlib import Path

p = Path(__file__).resolve().parents[1] / 'data' / 'biomet_data.csv'
arr = np.loadtxt(p, delimiter=None)
print('shape', arr.shape)
print('dtype', arr.dtype)
print('first row length', len(arr[0]))
print('first 3 rows')
print(arr[:3, :5])
