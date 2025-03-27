import numpy as np


def load_csv(filename, sep=';', skiprows=0):
    return np.loadtxt(open(filename, "rb"), delimiter=sep, skiprows=skiprows)

def load_cross_section(filename):
    """Takes a text file with two columns of numbers and return two arrays : one for each column"""
    data_cs = load_csv(filename)
    return data_cs[:,0], data_cs[:,1]


if __name__ == "__main__":
    e_r,cs_r=load_cross_section('..\cross_sections\Xe\exc_Xe.csv')
    print(e_r)
    print(cs_r)
    print("done")