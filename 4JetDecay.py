##RAMBO Momentum Generator
import numpy as np
import matrix2py
from tqdm import tqdm
import pandas as pd
import sys


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

    q = np.array([q_0, q_1, q_2, q_3])
    Q = np.sum(q, axis=1)
    M = np.sqrt(minkowski_product(Q, Q))
    b = - Q[1:]/M
    x = 1/M
    gamma = np.sqrt(1 + dot_product(b,b))
    a = 1/(1+gamma)

    p_0 = x*(gamma*q_0 + dot_product(q[1:],b[:,None]))
    p_123 = x*np.add(q[1:], np.outer(b, q[0] + a*dot_product(q[1:],b[:,None])))
    
    p = np.transpose(np.array([p_0, p_123[0], p_123[1], p_123[2]]))
    return p

def sing_event(CM, s):
    p = (CM**2 - s)/(2*CM) #Modulus of p_1
    E = np.sqrt(mod_p4**2 + s) #Energy of p234
    BOOST = [[gamma, 0, 0, -beta*gamma],
            [-beta*gamma*sin(theta), 0, 0, gamma*sin(theta)],
            [0, 0, 1, 0],
            [-beta*gamma*cos(theta), 0, 0, gamma*cos(theta)]]

    #Generate one full set of momenta and matrix element
    ##Incoming Momenta
    p_a = np.array([np.sqrt(CM), 0, 0, np.sqrt(CM)])/2
    p_b = np.array([np.sqrt(CM), 0, 0, -np.sqrt(CM)])/2
    
    ##Subset momenta. (s_234 small)
    #Momentum in s frame.
    mom_s = rambo(3)*s
    #Boost to CM frame.
    mom_cm = np.multiply(BOOST, mom_s, axis = 1)

    #Final parton. 
    p_1 = [p, p*cos(theta), 0, p*sin(theta)]
    
    me = matrix2py.get_value(np.transpose(np.concatenate(([p_a, p_b, p_1], mom))),alphas,nhel) #Matrix element calculation
    
    return (me, np.concatenate([p_1,mom]))

##Initital variables:
CM = 1000000 #Center of mass energy
n_jet = 4 #Number of jets
matrix2py.initialisemodel('../../Cards/param_card.dat')
alphas = 0.13
nhel = -1 # means sum over all helicity       
    
    
def genDataCSV(n_processes):
    ###Make Data
    mom_f=open('LO_mom_{}jet_{}.csv'.format(n_jet, n_process), 'ab')
    me_f=open('LO_me_{}jet_{}.csv'.format(n_jet, n_process), 'ab')
    for i in tqdm(range(n_process)):
        me, mom = sing_event(CM, n_jet)
        np.savetxt(mom_f, [np.ravel(mom)])
        np.savetxt(me_f, [me])
    me_f.close()
    mom_f.close()
    
def genDataNPY(n_processes):
    me = np.zeros(n_processes)
    mom = np.zeros((n_processes, n_jet, 4))
    for i, s in tqdm(enumerate(np.logspace(10**(-6),10**(1), 10))):
        me[i], mom[i] = sing_event(CM, s)
    np.save('LO_mom_{}jet_{}'.format(n_jet, n_processes), mom)
    np.save('LO_me_{}jet_{}'.format(n_jet, n_processes), me)

genDataNPY(int(sys.argv[1])) ##Enter number of datapoints when calling code (ie python GenDataLO.py 100000)
              
