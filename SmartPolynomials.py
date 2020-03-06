import numpy as np
from itertools import combinations_with_replacement as combs
from itertools import combinations
import DataPreprocessing
from functools import reduce
from sklearn.preprocessing import PolynomialFeatures


def smartPolynomialFeatures(n, mandel_str, mom):
    ##All possible terms in a Minkowski product of momenta.
    poly2 = np.array([np.multiply(mom[:,p,i],mom[:,q,i]) #Multiply components of 4-momenta in Minkowski product format
                     for i in range(4) for p in range(n) for q in range(p,n)]) #All combinations of outgoing 4-momenta, for each component
                
    poly2 = np.append(poly2,[mom[:,p,0] for p in range(n)],axis=0) #Minkowski product with incoming particles
    poly2 = np.append(poly2,[mom[:,p,3] for p in range(n)],axis=0)
    

    #All possible terms from a product of Minkowski products.
    poly = np.array([np.prod(comb, axis=0) for comb in combs(poly2, 2)])
    
    mandel = DataPreprocessing.mandel_creation(mandel_str[:4], mom)
    mandel = np.append(mandel, np.square(DataPreprocessing.mandel_creation(mandel_str[4:], mom)), axis=0)
    #Products of Mandelstam variables appearing in the formula.
        

    excluded_mandel = [(0,4),(0,6),(1,4),(1,7),(2,5),(2,6),(3,5),(3,7)]
    sets_mandel = np.array([reduce(np.multiply, [mandel[k] for k in range(8) if i!=k and j!=k]) 
                            for i,j in excluded_mandel])

 
    ##Multiply all poly4 by mandel variables.
    variables = np.array([np.multiply(p,q) for p in poly for q in sets_mandel])
    ##Return.
    return variables.T

##Minkowski product of 4-vectors p1, p2.
def m_prod_arr(p1, p2):
    return np.multiply(p1[:,0], p2[:,0]) - np.sum(np.multiply(p1[:,1:], p2[:,1:]), axis=1)


def smarterPolynomialFeatures(n, mom):
    p_a = np.array([1000, 0, 0, 1000])/2
    p_b = np.array([1000, 0, 0, -1000])/2
    
    
    mandel_strings = ['1,4','1,5','2,3','2,4','2,5','3,4','3,5','4,5']
    full_mom = np.insert(np.insert(mom, 0, p_a, axis=1), 0, p_b, axis=1)
    all_mandel = DataPreprocessing.mandel_creation(mandel_strings, full_mom)
    
    poly_features = PolynomialFeatures(degree=2)
    poly = poly_features.fit_transform(all_mandel.T)
    
    return poly
   

def allPolynomialFeatures(n, mom):
    mandel_str = ['1,3','2,3','1,4','2,4','1,2,3','1,2,4','1,3,4','2,3,4']
    mandel = DataPreprocessing.mandel_creation(mandel_str, mom)
    #Products of Mandelstam variables appearing in the formula.
    
    poly_features = PolynomialFeatures(degree=4)
    poly = poly_features.fit_transform(np.array([np.ndarray.flatten(np.array(element)) for element in mom[:,1:]]))

    
    excluded_mandel = [(0,4),(0,6),(1,4),(1,7),(2,5),(2,6),(3,5),(3,7)]
    sets_mandel = np.array([reduce(np.multiply, [mandel[k] for k in range(8) if i!=k and j!=k]) 
                            for i,j in excluded_mandel])

 
    ##Multiply all poly4 by mandel variables.
    variables = np.array([np.multiply(p,q) for p in poly.T for q in sets_mandel])
    ##Return.
    return variables.T

def somePolynomialFeatures(n, mom):
    mandel_str = ['1,3','2,3','1,4','2,4','1,2,3','1,2,4','1,3,4','2,3,4']
    mandel = DataPreprocessing.mandel_creation(mandel_str, mom)
    #mandel = np.append(mandel, np.square(DataPreprocessing.mandel_creation(mandel_str[4:], mom)), axis=0)
    #Products of Mandelstam variables appearing in the formula.
    
    mom_reduced = mom[:,1:]
    poly_features = PolynomialFeatures(degree=4)
    poly = poly_features.fit_transform(np.array([np.ndarray.flatten(np.array(element)) for element in mom_reduced]))

    
    excluded_mandel = [(0,4),(0,6),(1,4),(1,7),(2,5),(2,6),(3,5),(3,7)]
    #excluded_mandel = [(0,4),(0,5),(0,6),(0,7),(1,4),(1,5),(1,6),(1,7),(2,4),(2,5),(2,6),(2,7),(3,4),(3,5),(3,6),(3,7)]
    sets_mandel = np.array([reduce(np.multiply, [mandel[k] for k in range(8) if i!=k and j!=k]) 
                            for i,j in excluded_mandel])

 
    ##Multiply all poly4 by mandel variables.
    variables = np.array([np.multiply(p,q) for p in poly.T for q in sets_mandel])
    ##Return.
    return variables.T


'''mandel_str = ['1,3','2,3','1,4','2,4','1,2,3','1,2,4','1,3,4','2,3,4']


mom = np.array([[[ 1, 0, 0, 0],
                [ 1, 0, 0, 0],
                [ 0, 0, 0, 0]],
                [[ 1, 0, 0, 0],
                [ 1, 0, 0, 0],
                [ 0, 0, 0, 0]]])

mom = np.array([[[ 367.90382384,  226.34818408,  -44.32798829, -286.62650368],
  [ 132.79512135,  119.94579358,   32.63943529,  -46.71421754],
  [ 260.69994828, -203.11083547,   64.83164591,  150.02436215],
  [ 238.60110653, -143.1831422,   -53.14309291,  183.31635907]],

 [[ 124.14103209,   17.72180863,  -89.01124453,   84.69906548],
  [ 257.22476293,  -84.94043987, -237.91314289,   48.44622568],
  [ 328.6917174,  -156.32153416,  171.4752627,  -232.80476225],
  [ 289.94248758,  223.5401654,   155.44912472,   99.65947109]]])

print(smartPolynomialFeatures(3, mandel_str, mom).shape)



    #mandel_str = reduce(np.append, *[map(','.join,list(combinations(map(str,range(1,n+1)), i))) #Properly formatted
    #                for i in range(2,n)])  #All combinations of Mandelstam variables
    #print(mandel_str)'''