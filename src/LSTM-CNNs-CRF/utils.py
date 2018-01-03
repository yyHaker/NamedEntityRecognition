# -*- coding: utf-8 -*-
import pickle


def save_dict_to_file(dict, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(dict, f)


def load_dict_from_file(file_path):
    dict = {}
    with open(file_path, 'rb') as f:
        dict = pickle.load(f)
    return dict


if __name__ == "__main__":
    # a = {"accuracy": [1, 2, 3, 4], 'size': [1, 2, 34, 6]}
    # save_dict_to_file(a, 'result.pkl')
    print("DONE")
    b = load_dict_from_file('result.pkl')
    print(b)
