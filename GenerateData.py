##RAMBO Momentum Generator
import numpy as np
import matrix2py
from tqdm import tqdm
import pandas as pd


def minkowski_product(p1, p2):
    #Minkowski product of two 4-vectors
    return np.sum(p1[0]*p2[0] - p1[1]*p2[1] - p1[2]*p2[2] - p1[3]*p2[3])

def dot_product(v1, v2):
    #Dot product of two vectors
    return np.sum(np.multiply(v1, v2), axis=0)

def rambo(n = 5):
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

def sing_event(CM, n):
    #Generate one full set of momenta and matrix element
    p_a = np.array([np.sqrt(CM), 0, 0, np.sqrt(CM)])
    p_b = np.array([np.sqrt(CM), 0, 0, -np.sqrt(CM)])
    
    mom = rambo(n)*CM #Output momenta
    me = matrix2py.get_value(np.transpose(np.concatenate(([p_a, p_b], mom)))) #Matrix element calculation
   
    return (me, mom)

##Initital variables:
CM = 1000000 #Center of mass energy
n_process = 1000000 #Number of phase space points to generate
n_jet = 3 #Number of jets
matrix2py.initialisemodel('../../Cards/param_card.dat')
alphas = 0.13
nhel = -1 # means sum over all helicity       

###Make Data
mom_f=open('me_{}jet_{}'.format(n_jet, n_process), 'ab')
me_f=open('mom_{}jet_{}'.format(n_jet, n_process), 'ab')
for i in range(n_processes):
    me, mom = sing_event(CM, n_jet)
    np.savetxt(mom_f, mom)
    np.savetxt(me_f, me)
me_f.close()
mom_f.close()
    
              