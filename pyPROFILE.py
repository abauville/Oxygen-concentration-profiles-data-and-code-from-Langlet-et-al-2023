#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 11:23:03 2021

@author: abauville


Projet avec Dewi: optimization de la productivite en O2 des organismes en fonction de la profondeur
d'apres le papier de Berg et al 1998
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
import scipy.optimize as opt
from scipy.interpolate import interp1d
import pandas as pd






def compute_C(R, BC,  X, PA, K):
    # Solves the system of equations following eq. (1) to (10) of Berg et al., 1998

    # Miscellaneous
    # ===================
    Dx = X[1] - X[0]
    n = len(PA)

    # Boundary conditions
    # ===================
    # Set Boundary conditions
    BC_type = BC[0]
    BC_val1 = BC[1]
    BC_val2 = BC[2]
    C0      = BC[3]
    
    top_BC_type,bot_BC_type, top_BC_val, bot_BC_val = set_BC(BC_type, BC_val1, BC_val2, Dx, R)

    # Initialize parameters array
    # ===================
    AA = np.ones(n)
    BB = np.ones(n)
    CC = np.ones(n)
    DD = np.zeros(n)

    # Fill the system of equations
    # ===================
    I = slice(1,n-1)
    Im1 = slice(0,n-2)
    Ip1 = slice(2,n)

    AA[I] = 1.0/Dx * (2.0*K[Im1] * K[I  ]) / (Dx*K[I  ] + Dx*K[Im1])
    CC[I] = 1.0/Dx * (2.0*K[I  ] * K[Ip1]) / (Dx*K[Ip1] + Dx*K[I  ])
    BB[I] = -AA[I] - CC[I] - PA[I]
    DD[I] = -PA[I] *C0 - R[I]

    
    # Apply BC
    # ===================
    if top_BC_type==0:
        # Dirichlet top
        BB[0] = 1.0
        CC[0] = 0.0
        DD[0] = top_BC_val
    elif top_BC_type==1:
        # Neumann top
        DD[0] = top_BC_val
        BB[0] = (K[1]) / (Dx/1.) # /!\ the paper gives a factor 1/2, but empirically we saw that factor 1 yields reasonnable resutls, and 1/2 yields wrong value on the other boundary (e.g. when asking C=0, Flux=0 at the bottom this gives proper results)
        CC[0] = - (K[1] ) / (Dx/1.)
    else:
        raise ValueError('Unknwon top_BC_type')

    if bot_BC_type==0:
        # Dirichlet bottom
        BB[-1] = 1.0
        AA[-1] = 0.0
        DD[-1] = bot_BC_val
    elif bot_BC_type==1:
        # Neumann bottom
        DD[-1] = bot_BC_val
        BB[-1] = (K[-2]) / (Dx/1.)
        AA[-1] = -(K[-2] ) / (Dx/1.)
    else:
        raise ValueError('Unknwon bot_BC_type')
    
    
    
    A_mat = diags(AA[1:], -1, shape=(n,n)) + diags(BB, 0, shape=(n,n)) + diags(CC[:-1],1, shape=(n,n))
    RHS = DD.copy()

#     # Apply BC
#     # ===================
#     if top_BC_type==0:
#         # Dirichlet top
#         A_mat[0,0] = 1.0
#         A_mat[0,1] = 0.0
#         RHS[0] = top_BC_val
#     elif top_BC_type==1:
#         # Neumann top
#         BB0 = (K[1]) / (Dx/1.) # /!\ the paper gives a factor 1/2, but empirically we saw that factor 1 yields reasonnable resutls, and 1/2 yields wrong value on the other boundary (e.g. when asking C=0, Flux=0 at the bottom this gives proper )
#         CC0 = - (K[1] ) / (Dx/1.)
#         RHS[0] = top_BC_val
#         A_mat[0,0] = BB0
#         A_mat[0,1] = CC0
#     else:
#         raise ValueError('Unknwon top_BC_type')

#     if bot_BC_type==0:
#         # Dirichlet bottom
#         A_mat[-1,-1] = 1.0
#         A_mat[-1,-2] = 0.0
#         RHS[-1] = bot_BC_val
#     elif bot_BC_type==1:
#         # Neumann bottom
#         AA0 = -(K[-2] ) / (Dx/1.)
#         BB0 =  (K[-2]) / (Dx/1.)
#         RHS[-1] = bot_BC_val
#         A_mat[-1,-1] = BB0
#         A_mat[-1,-2] = AA0
#     else:
#         raise ValueError('Unknwon bot_BC_type')


    # Solve the system of equations for C
    # ===================
    C = spsolve(csr_matrix(A_mat), RHS)
    return C




def set_BC(BC_type, BC_val1, BC_val2, Dx, R):
    # Type of boundary conditions (1:t=C b=C, 2:t=C t=F, 3:b=C b=F 4:t=C b=F 5:t=F b=C)
    # Set boundary conditions
    if BC_type == 1:
        top_BC_type = 0 # 0: given concentration, 1:given flux
        bot_BC_type = 0 # 0: given concentration, 1:given flux
        top_BC_val = BC_val1
        bot_BC_val = BC_val2

    elif BC_type == 2:
        top_BC_type = 0 # 0: given concentration, 1:given flux
        bot_BC_type = 1 # 0: given concentration, 1:given flux
        top_BC_val = BC_val1
        bot_BC_val = np.sum(-R*Dx) - BC_val2

    elif BC_type == 3:
        top_BC_type = 1 # 0: given concentration, 1:given flux
        bot_BC_type = 0 # 0: given concentration, 1:given flux

        top_BC_val = np.sum(-R*Dx) - BC_val2
#         top_BC_val = np.sum(-(R[:-1]+R[1:])/2.0*Dx) - BC_val2
#         top_BC_val += 0.02

#         top_BC_val = np.sum(-(R[:-1]+R[1:])/2.0*Dx)# - BC_val2
        bot_BC_val = 0.0#BC_val1
#         print(top_BC_val)

    elif BC_type == 4:
        top_BC_type = 0 # 0: given concentration, 1:given flux
        bot_BC_type = 1 # 0: given concentration, 1:given flux

        top_BC_val = BC_val1
        bot_BC_val = BC_val2

    elif BC_type == 5:
        top_BC_type = 1 # 0: given concentration, 1:given flux
        bot_BC_type = 0 # 0: given concentration, 1:given flux

        top_BC_val = BC_val1
        bot_BC_val = BC_val2
    return (top_BC_type,bot_BC_type,
             top_BC_val, bot_BC_val)



def SSE(R_vals, C_GT, X, PA, K, BC):
    x_R = np.linspace(X[0], X[-1], len(R_vals))
    f = interp1d(x_R, R_vals, kind='linear')
    R_interp = f(X)

    SSE = 0
    for i in range(C_GT.shape[0]):
        gt = C_GT[i,:]
        BC = (1, gt[0], gt[-1])
        SSE += np.sum((compute_C(R_interp, BC,  X, PA, K)-gt)**2)
    return SSE


def model_profile(data_file, n_r=None):
    """Compute the model profile of concentration and optimize the value of net rate of production (R)

Parameters
----------
data_file: str
    name of a CSV file with the data. See example for the format
n_r: scalar, default: same as number of input data points
    number of points at which to compute R.

Returns
----------
df: pandas DataFrame
    contains initial data, model data (C_model) and optimal profile of R (R).
"""
    # Read constants
    # =================
    df_constants = pd.read_csv(data_file,skiprows=1, nrows=2,header=None,usecols=[0,1],index_col=0)
    D = df_constants.loc['D'].values[0]
    x0 = df_constants.loc['x0'].values[0]

    # Read data table
    # =================
    df = pd.read_csv(data_file, sep=',', skiprows=4)
    df = df[df['X']>=x0]
    X = df['X'].values
    phi = df['PHI'].values
    alpha = df['ALPHA'].values
    Db = df['DB'].values
    Ds = phi**2 * D

    K = phi*(Ds+Db) # Diffusivity
    PA = phi*alpha

    if n_r == None:
        n_r = len(K)

    # Extract the multiple concentration profiles
    # =================
    # note: GT stands for ground truth (i.e. "real data", not model)
    C_columns = [col for col in df.columns if col[0]=='C']
    C_GT = df[C_columns].values.T

    # Optimize R
    # =================
    R_vals_0 = np.zeros(n_r)
    opt_result = opt.minimize(SSE,R_vals_0,args=(C_GT, X, PA, K, BC))
    R_opt = opt_result.x

    # Add model data to the dataframe and return
    # =================
    x_R = np.linspace(X[0], X[-1], len(R_opt))
    f = interp1d(x_R, R_opt, kind='linear')
    R_interp = f(X)
    df['R'] = R_interp
    BC = (1, np.mean(C_GT[:,0]), np.mean(C_GT[:,-1]))
    df['C_model'] = compute_C(R_interp,BC, X, PA, K)
    return df

