import numpy as np
from itertools import combinations_with_replacement as combs
from itertools import combinations
import DataPreprocessing
from functools import reduce
from sklearn.preprocessing import PolynomialFeatures


def uuuxuxFeatures(mom):
    #####DENOMINATOR
    mandel_str_den = ['1,3', '1,4', '2,3', '2,4', '1,2,3', '1,2,4', '1,3,4', '2,3,4']
    mandel_den = DataPreprocessing.mandel_creation(mandel_str_den, mom)
    
    denom_parts = [(0,4),(0,6),(1,5),(1,6),(2,4),(2,7),(3,5),(3,7)]
    combs_index = list(combs(range(8), 2))
    excluded_combs = [(*denom_parts[a], *denom_parts[b]) for a, b in combs_index]

    
    mandel_powers = lambda excl : [np.power(mandel,2-excl.count(i)) for i, mandel in enumerate(mandel_den)]
    uncancelled_denom = np.array([reduce(np.multiply, mandel_powers(excl)) for excl in excluded_combs])
    
    
    #####NUMERATOR
    p_a = np.array([1000, 0, 0, 1000])/2
    p_b = np.array([1000, 0, 0, -1000])/2
    
    
    mandel_str_num = ['1,4','1,5','2,3','2,4','2,5','3,4','3,5','4,5']
    full_mom = np.insert(np.insert(mom, 0, p_a, axis=1), 0, p_b, axis=1)
    mandel_num = DataPreprocessing.mandel_creation(mandel_str_num, full_mom)
    
    num_features = PolynomialFeatures(degree=4)
    poly_numerator = num_features.fit_transform(mandel_num.T)
        
    
    ##Multiply all poly4 by mandel variables.
    variables = np.array([np.multiply(p,q) for p in poly_numerator.T for q in uncancelled_denom])
    
    return variables.T


def uuxgFeatures(mom):
    p_a = np.array([1000, 0, 0, 1000])/2
    p_b = np.array([1000, 0, 0, -1000])/2
    
    
    mandel_strings = ['1,3','1,4','2,3','2,4','3,4']
    full_mom = np.insert(np.insert(mom, 0, p_a, axis=1), 0, p_b, axis=1)
    all_mandel = DataPreprocessing.mandel_creation(mandel_strings, full_mom)
    
    poly_features = PolynomialFeatures(degree=2)
    poly = poly_features.fit_transform(all_mandel.T)
    
    return (poly)