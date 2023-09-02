import random
import pandas as pd
import numpy as np
import Covid19_model as CM
import matplotlib.pyplot as plt
import numpy as np
import shutil
import sys
import os.path
import math
import csv
from pyomo.environ import *
import time 
import pickle


def solve_pyomo(init_state, budget, para, H):
    p_ei  = para['pei']
    p_er  = para['per']
    p_ea  = 1-p_ei-p_er
    p_ai  = p_ei
    p_ar  = p_er
    p_irr = 1/8
    alpha  = 0.6
    beta_1 = para['beta1']
    beta_2 = para['beta2']
    cost_tr = 0.0977
    p_id  = 1.1/1000
    p_idd = 0.1221/1000
    p_srr = 0.7/3
    S = init_state[0]
    E = init_state[1]
    A = init_state[2]
    I = init_state[3]
    R = init_state[4]
    D = init_state[5]
    N = np.sum(init_state)
    K = H - 1
    w_max  = 1.0
    tr_max = 1.0
    v_max = 0.2/14
    M = budget
    
    m = ConcreteModel()
    m.k = RangeSet(0, K)
    m.t = RangeSet(0, K-1)
    m.N = N
    m.tri = Var(m.k, bounds=(0, 1)) 
    m.wi = Var(m.k, bounds=(0, 1))  
    m.v = Var(m.k, bounds=(0, 1))
    m.aq = Var(m.k, bounds=(0, 1))
    m.iq = Var(m.k, bounds=(0, 1))
    m.x_s = Var(m.k, domain = NonNegativeReals)
    m.x_e = Var(m.k, domain = NonNegativeReals)
    m.x_a = Var(m.k, domain = NonNegativeReals)
    m.x_i = Var(m.k, domain = NonNegativeReals)
    m.x_r = Var(m.k, domain = NonNegativeReals)
    m.x_d = Var(m.k, domain = NonNegativeReals)
    
    m.xs_c = Constraint(m.t, rule=lambda m, k: m.x_s[k+1] == m.x_s[k]
                    -(beta_1*(1-m.iq[k])*m.x_i[k]+beta_2*(m.x_e[k]+(1-m.aq[k])*m.x_a[k]))
                    *(1-m.v[k]*v_max)*m.x_s[k]/(m.x_s[k]+m.x_e[k]+m.x_a[k]+m.x_i[k]+m.x_r[k])
                    -p_srr*m.v[k]*v_max*m.x_s[k])

    m.xe_c = Constraint(m.t, rule=lambda m, k: m.x_e[k+1] == m.x_e[k]
                        +alpha*(beta_1*(1-m.iq[k])*m.x_i[k]+beta_2*(m.x_e[k]+(1-m.aq[k])*m.x_a[k]))
                        *(1-m.v[k]*v_max)*m.x_s[k]/(m.x_s[k]+m.x_e[k]+m.x_a[k]+m.x_i[k]+m.x_r[k])
                        -p_ei*m.x_e[k]
                        -p_er*m.x_e[k]
                        -m.wi[k]*p_ea*m.x_e[k])
                     
    m.xa_c = Constraint(m.t, rule=lambda m, k: m.x_a[k+1] == m.x_a[k]
                        +m.wi[k]*p_ea*m.x_e[k]
                        -p_ai*m.x_a[k]
                        -p_ar*m.x_a[k])
                                        
    m.xi_c = Constraint(m.t, rule=lambda m, k: m.x_i[k+1] == m.x_i[k]
                        +(1-alpha)*(beta_1*(1-m.iq[k])*m.x_i[k]+beta_2*(m.x_e[k]+(1-m.aq[k])*m.x_a[k]))
                        *(1-m.v[k]*v_max)*m.x_s[k]/(m.x_s[k]+m.x_e[k]+m.x_a[k]+m.x_i[k]+m.x_r[k])
                        +p_ei*m.x_e[k]
                        +p_ai*m.x_a[k]
                        -m.tri[k]*p_irr*m.x_i[k]
                        -(1-m.tri[k])*m.x_i[k]*p_id
                        -m.tri[k]*m.x_i[k]*p_idd)
                                                                                       
    m.xr_c = Constraint(m.t, rule=lambda m, k: m.x_r[k+1] == m.x_r[k]
                        +m.tri[k]*p_irr*m.x_i[k]
                        +p_er*m.x_e[k]
                        +p_ar*m.x_a[k]
                        +p_srr*m.v[k]*v_max*m.x_s[k])

    m.xd_c = Constraint(m.t, rule=lambda m, k: m.x_d[k+1] == m.x_d[k]
                        +(1-m.tri[k])*m.x_i[k]*p_id
                        +m.tri[k]*m.x_i[k]*p_idd)
    
    m.pc = ConstraintList()
    m.pc.add(m.x_s[0]==S)
    m.pc.add(m.x_e[0]==E)
    m.pc.add(m.x_a[0]==A)
    m.pc.add(m.x_i[0]==I)
    m.pc.add(m.x_r[0]==R)
    m.pc.add(m.x_d[0]==D)

    gamma = 1
    m.sumcost = sum(m.wi[k]*(0.000369*(m.wi[k]*(m.x_s[k]+m.x_a[k]+m.x_e[k]+m.x_r[k])/m.N)
                         *(m.wi[k]*(m.x_s[k]+m.x_e[k]+m.x_a[k]+m.x_r[k])/m.N) + 0.001057)
                         *(m.x_s[k]+m.x_e[k]+m.x_a[k]+m.x_r[k])
                    +0.0977*m.tri[k]*m.x_i[k]+m.x_s[k]
                    *m.v[k]*v_max*0.07 + m.x_i[k]*m.iq[k]*0.02 + m.x_a[k]*m.aq[k]*0.02 for k in m.t) 
    m.budget_c = Constraint(expr = m.sumcost <= M)

    m.suminfected = sum(2.6 * (beta_1*(1-m.iq[k])*m.x_i[k]+beta_2*(m.x_e[k]+(1-m.aq[k])*m.x_a[k]))
                        *(1-m.v[k]*v_max)*m.x_s[k]/(m.x_s[k]+m.x_e[k]+m.x_a[k]+m.x_i[k]+m.x_r[k])
                        +(m.x_d[k+1] - m.x_d[k]) * 60 for k in m.t)
    m.obj = Objective(expr = m.suminfected, sense = minimize)
    solver = SolverFactory('ipopt')
    solver.options['max_iter'] = 100000
    solver.solve(m,tee=False)
    k = np.array([k for k in m.k])
    tri = np.array([m.tri[k]() for k in m.k])
    wi = np.array([m.wi[k]() for k in m.k])
    vi = np.array([m.v[k]() for k in m.k])
    aq  = np.array([m.aq[k]() for k in m.k])
    iq  = np.array([m.iq[k]() for k in m.k])
    # return [vi[0], tri[0], aq[0], iq[0], wi[0]]
    return [vi, tri, aq, iq, wi]


def mpc(interval, H):
    init_state = np.array([100000, 20, 0, 0, 0, 0])
    N = np.sum(init_state)
    n_state = 6
    n_action = 2
    T = 50
    budget = 3.6e3
    #H = 5
    deter_para = dict(N=N, B=budget, T=T, alpha=0.6, v_max=0.2 / 14,
                      cost_ti=0.0977, cost_ta=0.02, cost_v=0.07, cost_poc_0=0.000369,
                      cost_poc_1=0.001057, pid=1.10 / 1000, psr=0.7 / 3,
                      pid_plus=0.1221 / 1000, pir=1 / 8)

    in_deter_para = dict(beta2=[0.78735 * (1-interval/2), 0.78735 * (1+interval/2)],
                         beta1=[0.15747 * (1-interval/2), 0.15747 * (1+interval/2)],
                         pei=[0.10714 * (1-interval/2), 0.10714 * (1+interval/2)],
                         per=[0.04545 * (1-interval/2), 0.04545 * (1+interval/2)])
    tic = time.time()
    result = []
    for s in range(10):
        env = CM.Env_model(init_state, deter_para, in_deter_para, 0)
        para_truth = pickle.load(open('para_truth' + str(round(interval * 100)), 'rb'))[s]
        env.set_para_truth(para_truth)
        budget = 3.6e3
        state, observation = init_state.copy(), init_state.copy()
        state_buffer = np.zeros([T, 6])
        new_buffer = np.zeros([T, 4])
        score, ni, ne = 0, 0, 0 
        done = False
        index = [0]

        for t in range(T):
            pset = env.para_sampling(100)
            para = pset[index[0]]
            state[state < 0] = 0
            if budget > 0:
                try:
                    aa = solve_pyomo(state, budget, para, H)
                    action = [aa[0][0], aa[1][0], aa[2][0], aa[3][0], aa[4][0]]
                except:
                    print()
            else:
                action =np.array([0, 0, 0, 0, 0]) 
            cost = env.cost_function(state, action)
            budget -=cost       
            next_state, obs, reward, cost, new_i, new_e, ck = env.online_state_estimate(state, observation, 
                                                                                       action, 1, 100, 1)
            index = ck.argsort()[0][:1]

            score += reward
            state = next_state.copy()
            observation = obs.copy()
            print(['time:', t, 'state:', state])
            print(['action:', action])
        result.append(score)

    print(result)
    print(np.mean(result))
    print(time.time() - tic)
    
def mpc_out(interval,out, H):
    init_state = np.array([100000, 20, 0, 0, 0, 0])
    N = np.sum(init_state)
    n_state = 6
    n_action = 2
    T = 50
    budget = 3.6e3
    
    deter_para = dict(N=N, B=budget, T=T, alpha=0.6, v_max=0.2 / 14,
                      cost_ti=0.0977, cost_ta=0.02, cost_v=0.07, cost_poc_0=0.000369,
                      cost_poc_1=0.001057, pid=1.10 / 1000, psr=0.7 / 3,
                      pid_plus=0.1221 / 1000, pir=1 / 8)

    in_deter_para = dict(beta2=[0.78735 * (1-interval/2), 0.78735 * (1+interval/2)],
                         beta1=[0.15747 * (1-interval/2), 0.15747 * (1+interval/2)],
                         pei=[0.10714 * (1-interval/2), 0.10714 * (1+interval/2)],
                         per=[0.04545 * (1-interval/2), 0.04545 * (1+interval/2)])

    tic = time.time()
    result = []
    action =np.array([1, 1, 1, 1, 1])
    for s in range(10):
        env = CM.Env_model(init_state, deter_para, in_deter_para, 0)
        para_truth = pickle.load(open('out_of_sample_' + str(round(interval * 100)) + '_' + str(round(out * 100)), 'rb'))[s]
        env.set_para_truth(para_truth)
        budget = 3.6e3
        state, observation = init_state.copy(), init_state.copy()
        state_buffer = np.zeros([T, 6])
        new_buffer = np.zeros([T, 4])
        score, ni, ne = 0, 0, 0 
        done = False
        index = [0]

        for t in range(T):
            pset = env.para_sampling(100)
            para = pset[index[0]]
            state[state < 0] = 0
            if budget > 0:
                try:
                    aa = solve_pyomo(state, budget, para, H)
                    action = [aa[0][0], aa[1][0], aa[2][0], aa[3][0], aa[4][0]]
                except:
                    print()
            else:
                action =np.array([0, 0, 0, 0, 0]) 
            cost = env.cost_function(state, action)
            budget -=cost       
            next_state, obs, reward, cost, new_i, new_e, ck = env.online_state_estimate(state, observation, 
                                                                                       action, 1, 100, 1)
            index = ck.argsort()[0][:1]

            score += reward
            state = next_state.copy()
            observation = obs.copy()
            print(['time:', t, 'state:', state])
            print(['action:', action])
        result.append(score)

    print(result)
    print(np.mean(result))
    
    