import numpy as np
from functools import reduce

##Minkowski product of 4-vectors p1, p2.
def m_prod_arr(p1, p2):
    return np.multiply(p1[:,0], p2[:,0]) - np.sum(np.multiply(p1[:,1:], p2[:,1:]), axis=1)

##Create mandelstam variables causing IR divergences. combs_str = ['1,2,3', '2,3', '4,5',...] -> mandel_vars = [s_123, s_23, s_45,...]
def mandel_creation_csv(combs_str, mom_flat):
    mandel_vars = []
    for comb in combs_str:
        p = np.sum(np.array([mom_flat[:, (int(i) - 1)*4: int(i)*4] for i in comb.split(',')]), axis=0) 
        mandel_vars.append(m_prod_arr(p, p)) 
    return np.array(mandel_vars)

##NPY mandel creation (mom still structured)
def mandel_creation(combs_str, mom):
    mandel_vars = []
    for comb in combs_str:
        p = np.sum(np.array([mom[:,int(i)-1] for i in comb.split(',')]), axis=0)
        mandel_vars.append(m_prod_arr(p,p))
    return np.array(mandel_vars)

def npy(me_filename, mom_filename, combs_str, frac=1):
    ##combs_str is the xyz in s_xyz... which the matrix element is multiplied by, to remove IR infinites.
    ##Data Aquisition
    me_raw = np.load(me_filename, allow_pickle=True) #Matrix elements
    #me_raw = np.ones(len(me_raw))
    mom_raw = np.load(mom_filename, allow_pickle=True, encoding='bytes') #4-momenta of inputs
    ##Obtain fraction of data
    me=me_raw[:int(frac*len(me_raw))]
    mom=mom_raw[:int(frac*len(mom_raw))]

    ##Reformat Matrix Element (remove divergent behaviour)
    if len(combs_str) > 0:
        mandel_vars = reduce(np.multiply, mandel_creation(combs_str, mom))
        me = np.multiply(me, mandel_vars)

    return(me, mom)

def csv(me_filename, mom_filename, combs_str, com_energy, frac=1):
    ##Data Aquisition
    me_raw = np.loadtxt(me_filename) #Matrix elements
    mom_raw = np.loadtxt(mom_filename) #4-momenta of inputs

    ##Obtain fraction of data
    me=me_raw[:int(frac*len(me_raw))]
    mom=mom_raw[:int(frac*len(mom_raw))]

    ##Reformat Matrix Element (remove divergent behaviour)
    if len(combs_str) > 0:
        mandel_vars = reduce(np.multiply, mandel_creation(combs_str, mom))/com_energy**(2*len(combs_str)) 
        me = np.multiply(me, mandel_vars)
    
    return(me, mom)

def npyNLO(me_filename, mom_filename, combs_str, frac=1):
    ##combs_str is the xyz in s_xyz... which the matrix element is multiplied by, to remove IR infinites.
    ##Data Aquisition
    me_raw = np.load(me_filename, allow_pickle=True) #Matrix elements
    mom_raw = np.load(mom_filename, allow_pickle=True, encoding='bytes') #4-momenta of inputs
    ##Obtain fraction of data
    me=me_raw[:int(frac*len(me_raw))]
    mom=mom_raw[:int(frac*len(mom_raw))]
    

    born_me, loop_me, r1, r2 = np.transpose(me)
    ##Reformat Matrix Element (remove divergent behaviour)
    if len(combs_str) > 0:
        mandel_vars = reduce(np.multiply, mandel_creation(combs_str, mom)) 
        born_me = np.multiply(born_me, mandel_vars)
        loop_me = np.multiply(loop_me, mandel_vars)

    return(born_me, loop_me, r1, r2, mom)