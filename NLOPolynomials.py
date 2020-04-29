import numpy as np
import scipy
from itertools import combinations_with_replacement as combs
from itertools import combinations
import DataPreprocessing
from functools import reduce
from sklearn.preprocessing import PolynomialFeatures


def uuxgFinFeatures(mom):
    degree2 = PolynomialFeatures(degree=2)
    degree4 = PolynomialFeatures(degree=4)
    mandel = DataPreprocessing.mandel_creation(['1,3','2,3','1,2'], mom)
    S = 1000000
    
    ##BORN FEATURES
    p_a = np.array([1000, 0, 0, 1000])/2
    p_b = np.array([1000, 0, 0, -1000])/2
    
    mandel_strings = ['1,3','1,4','2,3','2,4']
    full_mom = np.insert(np.insert(mom, 0, p_a, axis=1), 0, p_b, axis=1)
    all_mandel = DataPreprocessing.mandel_creation(mandel_strings, full_mom)
    
    poly = degree2.fit_transform(all_mandel.T)
    
    ##LOGS
    poly_logs = degree2.fit_transform(np.log(mandel).T)
    poly2 = np.array([np.multiply(p,q) for p in poly.T for q in poly_logs.T])
    
    
    dialogs = scipy.special.spence(np.divide(S,mandel))
    poly_dialogs = degree2.fit_transform(dialogs.T)
    poly = np.array([np.multiply(p,q) for p in poly2 for q in poly_dialogs.T])
    
    return poly.T

def uuxgr1Features(mom):
    degree1 = PolynomialFeatures(degree=1)
    degree2 = PolynomialFeatures(degree=2)
    S = 1000000
    mandel_str = ['2,3','1,3','1,2']
    mandel = DataPreprocessing.mandel_creation(mandel_str, mom)
    
    ###BORN FEATURES
    p_a = np.array([1000, 0, 0, 1000])/2
    p_b = np.array([1000, 0, 0, -1000])/2
    
    mandel_strings = ['1,3','1,4','2,3','2,4']
    full_mom = np.insert(np.insert(mom, 0, p_a, axis=1), 0, p_b, axis=1)
    all_mandel = DataPreprocessing.mandel_creation(mandel_strings, full_mom)
    
    poly_features = PolynomialFeatures(degree=2)
    poly = poly_features.fit_transform(all_mandel.T)
    
    ##LOGS
    poly_logs = degree2.fit_transform(np.log(mandel).T)
    poly2 = np.array([np.multiply(p,q) for p in poly.T for q in poly_logs.T])
    
    return poly2.T

def uuuxuxFinFeatures(mom):
    degree1 = PolynomialFeatures(degree=1)
    degree2 = PolynomialFeatures(degree=2)
    mandel = DataPreprocessing.mandel_creation(['1,3','2,3','1,4','2,4'], mom)
    S = 1000000
    
    #####DENOMINATOR
    mandel_str_den = ['1,3', '1,4', '2,3', '2,4', '1,2,3', '1,2,4', '1,3,4', '2,3,4']
    mandel_den = DataPreprocessing.mandel_creation(mandel_str_den, mom)
    
    denom_parts = [(0,4),(0,6),(1,5),(1,6),(2,4),(2,7),(3,5),(3,7)]
    combs_index = list(combs(range(8), 2))
    excluded_combs = [(*denom_parts[a], *denom_parts[b]) for a, b in combs_index]
    
    mandel_powers = lambda excl : [np.power(mandel,2-excl.count(i)) for i, mandel in enumerate(mandel_den)]
    uncancelled_denom = np.array([reduce(np.multiply, mandel_powers(excl)) for excl in excluded_combs])
    
    #####NUMERATOR
    ##BORN
    p_a = np.array([1000, 0, 0, 1000])/2
    p_b = np.array([1000, 0, 0, -1000])/2
    
    
    mandel_str_num = ['1,4','1,5','2,3','2,4','2,5','3,4','3,5','4,5']
    full_mom = np.insert(np.insert(mom, 0, p_a, axis=1), 0, p_b, axis=1)
    mandel_num = DataPreprocessing.mandel_creation(mandel_str_num, full_mom)
    
    num_features = PolynomialFeatures(degree=4)
    poly_numerator = num_features.fit_transform(mandel_num.T)
    
    poly = np.array([np.multiply(p,q) for p in poly_numerator.T for q in uncancelled_denom])
    
    ##LOGS
    poly_logs = degree2.fit_transform(np.log(mandel).T)
    poly = np.array([np.multiply(p,q) for p in poly.T for q in poly_logs.T])
    
    
    dialogs = scipy.special.spence(np.divide(S,mandel))
    poly_dialogs = degree2.fit_transform(dialogs.T)
    poly = np.array([np.multiply(p,q) for p in poly for q in poly_dialogs.T])
    
    return poly.T