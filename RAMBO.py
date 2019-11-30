##RAMBO Momentum Generator
import numpy as np

def minkowski_product(p1, p2):
    return np.sum(p1[0]*p2[0] - p1[1]*p2[1] - p1[2]*p2[2] - p1[3]*p2[3])

def dot_product(v1, v2):
    return np.sum(np.multiply(v1, v2), axis=0)

def generate_momenta(n = 4):
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