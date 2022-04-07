import numpy as np 
from typing import Tuple 

def simulate(d: int, n: int, p: float) -> Tuple[int, float]: 
    '''Computes number of tests and efficiency
    Parameters: 
        d(int): number of samples in each group
        n(int): number of groups 
        p(float): prevalence 
    Returns: 
        (int, float): number of tests, efficiency
    '''
    samples = np.random.rand(d, n) <= p 
    pos_groups = np.any(samples, axis=0) 
    num_tests = n + d * np.sum(pos_groups)  
    efficiency = (d * n) / num_tests  
    return num_tests, efficiency 


def analytical_efficiency(d: int, p: float) -> float:  
    '''Analytically computes efficiency 
    Parameters: 
        d(int): number of samples in each group
        p(float): prevalence 
    Returns: 
        float: efficiency
    '''    
    pos_group_prob = 1 - (1 - p) ** d
    efficiency = d / (pos_group_prob * d + 1)
    return efficiency 


def find_max_efficiency(d_max: int, n: int, p: float) -> Tuple[int, float]:  
    '''Finds maximal efficiency 
    Parameters: 
        d_max(int): defines search space, 2..d_max 
        n(int): number of groups 
        p(float): prevalence 
    Returns: 
        (int, float): optimal d, maximal efficiency
    '''        
    efficiencies = ((d, simulate(d, n, p)[1]) for d in range(2, d_max + 1)) 
    return max(efficiencies, key=lambda v: v[1])


def find_max_analytical_efficiency(d_max: int, p: float) -> Tuple[int, float]: 
    '''Finds maximal efficiency from analytical formula 
    Parameters: 
        d_max(int): defines search space, 2..d_max 
        p(float): prevalence 
    Returns: 
        (int, float): optimal d, maximal efficiency
    '''        
    efficiencies = ((d, analytical_efficiency(d, p)) for d in range(2, d_max + 1)) 
    return max(efficiencies, key=lambda v: v[1])
