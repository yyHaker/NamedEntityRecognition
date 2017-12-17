# -*- coding: utf-8 -*-
chars2 = [[1, 2, 4], [6, 7, 4, 6, 1, 3, 4], [5, 6, 3, 5], [2]]
chars2_sorted = sorted(chars2, key=lambda p: len(p), reverse=True)
print(chars2_sorted)
d = {}  #
for i, ci in enumerate(chars2):
    for j, cj in enumerate(chars2_sorted):  # [[6, 7, 4, 6, 1, 3, 4], [5, 6, 3, 5], [1, 2, 4], [2]]
        if ci == cj:
            d[j] = i
            continue
print(d)