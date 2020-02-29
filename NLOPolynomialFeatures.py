import numpy as np
import scipy
from itertools import combinations_with_replacement as combs
from itertools import combinations
import DataPreprocessing
from functools import reduce
from sklearn.preprocessing import PolynomialFeatures


def finitePolynomialFeatures(mandel_str, mom, S):   
    mandel = DataPreprocessing.mandel_creation(mandel_str, mom)
    
    #Born level features
    degree2 = PolynomialFeatures(degree=2)
    poly_mom = degree2.fit_transform(np.array([np.ndarray.flatten(np.array(element)) for element in mom]))
    poly_logs = degree2.fit_transform(np.log(mandel).T)
    poly2 = np.array([np.multiply(p,q) for p in poly_mom.T for q in poly_logs.T])
    
    degree1 = PolynomialFeatures(degree=1)
    dialogs = scipy.special.spence(np.divide(S,mandel))
    poly_dialogs = degree1.fit_transform(dialogs.T)
    poly = np.array([np.multiply(p,q) for p in poly2 for q in poly_dialogs.T])
    
    #Products of Mandelstam variables appearing in the formula.

    #logs = np.square(np.log(mandel)) #Logarithmic terms
    #dialogs = scipy.special.spence(np.divide(S,mandel)) #Dialogarithmic terms
    #mandel_square = np.square(mandel)
    #numerator = np.append(mandel_square, np.append(logs, dialogs, axis=0),axis=0) #Combination of all terms.
    #numerator = np.append(numerator,np.ones((1,len(mom))),axis=0)

    ##Denominator is (S-S_13) & (S-S_23) & S_13 & S_23
    denom_a = np.multiply(S-mandel[0], S-mandel[1])
    denom_b = np.multiply(mandel[0], mandel[1])
    denominator = np.append([np.multiply(denom_b, S-mandel[0])], [np.multiply(denom_b, S-mandel[1])],axis=0)
    denominator = np.append(denominator, [denom_a], axis=0)
    
    ##Multiply all poly4 by mandel variables.
    variables = np.array([np.multiply(p,q) for p in poly for q in denominator])

    ##Return.
    return variables.T

def r1PolynomialFeatures(mandel_str, mom, S):
    mandel_str = ['2,3','1,3']
    mandel = DataPreprocessing.mandel_creation(mandel_str, mom)
    
    ##Numerator is polynomials degree 2, constants, and logs of S_13, S_23.
    features = PolynomialFeatures(degree=2)
    
    poly_mom = features.fit_transform(np.array([np.ndarray.flatten(np.array(element)) for element in mom]))
    poly_logs = features.fit_transform(np.log(mandel).T)
    poly = np.array([np.multiply(p,q) for p in poly_mom.T for q in poly_logs.T])
    
    
    ##Denominator is (S-S_13) & (S-S_23) & S_13 & S_23
    denom_a = np.multiply(S-mandel[0], S-mandel[1])
    denom_b = np.multiply(mandel[0], mandel[1])
    denominator = np.append([np.multiply(denom_b, S-mandel[0])], [np.multiply(denom_b, S-mandel[1])],axis=0)
    denominator = np.append(denominator, [denom_a], axis=0)
    
    ##Denominator Terms
    denom_mandel = np.append(mandel, S-mandel, axis = 0)
    divergences = PolynomialFeatures(degree=2)
    denominator = divergences.fit_transform(mandel.T)
    

    ##Multiply all poly4 by mandel variables.
    variables = np.array([np.multiply(p,q) for p in poly for q in denominator.T])
    ##Return.
    return variables.T

