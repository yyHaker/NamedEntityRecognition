# -*- coding: utf-8 -*-
import numpy as np
from utils import load_data_from_file
import matplotlib.pyplot as plt

f1_result = load_data_from_file('f1_result')
dev_f1 = f1_result["dev_f1"]
test_f1 = f1_result["test_f1"]

x_num = np.arange(1, len(dev_f1) + 1)


plt.figure()
plt.plot(x_num, dev_f1, label="dev_F1")
plt.plot(x_num, test_f1, color='red', linewidth=1.0, linestyle='--', label="test_F1")
plt.legend()
plt.xlabel("EPOCH")
plt.ylabel("F1")
plt.title("dev and test F1")
plt.show()
