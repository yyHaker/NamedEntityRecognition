# -*- coding: utf-8 -*-
import numpy as np
import pickle

# write the list data to the file
"""
a = np.array([1, 2, 3, 4, 5])
b = np.array([4, 5, 6, 7, 8])

with open('data.pkl', 'wb') as f:
    pickle.dump(a, f)
    pickle.dump(b, f)
"""
print("-"*30)

# read data
with open('data.pkl', 'rb') as f:
    a = pickle.load(f)
    print(a)
    print('-'*30)
with open('data.pkl', 'rb') as f:
    b = pickle.load(f)
    print(b)
    print("-"*30)

# read the data from file

with open('data.pkl', 'rb') as f:
    a = pickle.load(f)
    b = pickle.load(f)
    print("-"*10)
    print(a)
    print("-"*10)
    print(b)
    print(type(b))  # <class 'numpy.ndarray'>



