##RAMBO Momentum Generator
from __future__ import division 
import numpy as np
import matrix2py
from tqdm import tqdm
import pandas as pd
import sys

##TEST CODE
##Minkowski product of 4-vectors p1, p2.
def m_prod_arr(p1, p2):
    return np.multiply(p1[0], p2[0]) - np.sum(np.multiply(p1[1:], p2[1:]), axis=1)

def mandel_creation(mom, combs_str):
    mandel_vars=[]
    for comb in combs_str:
        p = np.sum(np.array([mom[int(i)-1] for i in comb.split(',')]), axis=0)
        mandel_vars.append(m_prod_arr(p,p))
    return np.array(mandel_vars, dtype=np.longdouble)



####


def minkowski_product(p1, p2):
    #Minkowski product of two 4-vectors
    return np.sum(p1[0]*p2[0] - p1[1]*p2[1] - p1[2]*p2[2] - p1[3]*p2[3])

def dot_product(v1, v2):
    #Dot product of two vectors
    return np.sum(np.multiply(v1, v2), axis=0)

def rambo(n = 4):
    #Random phase space generator RAMBO.
    rho_1, rho_2, rho_3, rho_4 = np.random.rand(4, n)

    c = 2*rho_1 - 1
    phi = 2*np.pi*rho_2

    q_0 = - np.log(np.multiply(rho_3,rho_4))
    q_1 = q_0*np.sqrt(1-c**2)*np.cos(phi)
    q_2 = q_0*np.sqrt(1-c**2)*np.sin(phi)
    q_3 = q_0*c

    q = np.array([q_0, q_1, q_2, q_3],dtype=np.longdouble)
    Q = np.sum(q, axis=1)
    M = np.sqrt(minkowski_product(Q, Q))
    b = - Q[1:]/M
    x = 1/M
    gamma = np.sqrt(1 + dot_product(b,b))
    a = 1/(1+gamma)

    p_0 = x*(gamma*q_0 + dot_product(q[1:],b[:,None]))
    p_123 = x*np.add(q[1:], np.outer(b, q[0] + a*dot_product(q[1:],b[:,None])))
    
    p = np.transpose(np.array([p_0, p_123[0], p_123[1], p_123[2]],dtype=np.longdouble))
    return p

def sing_event(CM, s):
    p = (CM**2 - s)/(2*CM) #Modulus of p_1
    E = np.sqrt(p**2 + s) #Energy of p234

    theta = 1
    gamma = E/s
    beta = p/E

    BOOST = np.array([[gamma, 0, 0, -beta*gamma],
            [-beta*gamma*np.sin(theta), 0, 0, gamma*np.sin(theta)],
            [0, 0, 1, 0],
            [-beta*gamma*np.cos(theta), 0, 0, gamma*np.cos(theta)]],dtype=np.longdouble)

    #Generate one full set of momenta and matrix element
    ##Incoming Momenta
    p_a = np.array([CM, 0, 0, CM],dtype=np.longdouble)/2
    p_b = np.array([CM, 0, 0, -CM],dtype=np.longdouble)/2
    
    ##Subset momenta. (s_234 small)
    #Momentum in s frame.
    mom_s = rambo(3)*s
    #print('s frame: {}'.format(np.sum(mom_s,axis=0)))
    #Boost to CM frame.
    mom_cm = np.array([np.dot(BOOST, mom_part) for mom_part in mom_s],dtype=np.longdouble)
    #print('cm frame: {}'.format(np.sum(mom_cm,axis=0)))
    #Final parton. 
    p_1 = np.array([p, p*np.sin(theta), 0, p*np.cos(theta)],dtype=np.longdouble)
    #print('p : {}'.format(p_1))

    me = matrix2py.get_value(np.transpose(np.concatenate(([p_a, p_b, p_1], mom_cm))),alphas,nhel) #Matrix element calculation
    
    return (me, np.concatenate(([p_1],mom_cm)))

##Initital variables:
CM = 1000 #Center of mass energy
n_jet = 4 #Number of jets
matrix2py.initialisemodel('../../Cards/param_card.dat')
alphas = 0.13
nhel = -1 # means sum over all helicity       
    
    
def genDataNPY(n_processes):
    me = np.zeros(n_processes,dtype=np.longdouble)
    mom = np.zeros((n_processes, n_jet, 4),dtype=np.longdouble)
    for i, s in tqdm(enumerate(np.logspace(-10,-4, n_processes))):
        me[i], mom[i] = sing_event(CM, s)
        print('s : {}, me: {}, new_s : {}'.format(s,me[i], mandel_creation(mom[i], '2,3,4')))
        #print('total sum : {}'.format(np.sum(mom[i], axis=0)))
    np.save('LO_mom_{}jet_{}'.format(n_jet, n_processes), mom)
    np.save('LO_me_{}jet_{}'.format(n_jet, n_processes), me)

genDataNPY(int(sys.argv[1])) ##Enter number of datapoints when calling code (ie python GenDataLO.py 100000)
              
