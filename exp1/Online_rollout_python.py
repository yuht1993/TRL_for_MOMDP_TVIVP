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


def online_rollout(interval):
    init_state = np.array([100000, 20, 0, 0, 0, 0])
    N = np.sum(init_state)
    n_state = 6
    n_action = 2
    T = 50
    budget = 3.6e3 
    num_sample = 20000
    deter_para = dict(N=N, B=budget, T=T, alpha=0.6, v_max=0.2 / 14,
                      cost_ti=0.0977, cost_ta=0.02, cost_v=0.07, cost_poc_0=0.000369,
                      cost_poc_1=0.001057, pid=1.10 / 1000, psr=0.7 / 3,
                      pid_plus=0.1221 / 1000, pir=1 / 8)

    in_deter_para = dict(beta2=[0.78735 * (1-interval/2), 0.78735 * (1+interval/2)],
                         beta1=[0.15747 * (1-interval/2), 0.15747 * (1+interval/2)],
                         pei=[0.10714 * (1-interval/2), 0.10714 * (1+interval/2)],
                         per=[0.04545 * (1-interval/2), 0.04545 * (1+interval/2)])

    in_deter_truth = dict(beta1=0.15747, beta2=0.78735, pei=0.10714, per=0.04545)

    para_truth = {**deter_para, **in_deter_truth}
    random.seed(0)
    tic = time.time()
    result = []
    for s in range(10):
        env = CM.Env_model(init_state, deter_para, in_deter_para, 0)
        para_truth = pickle.load(open('para_truth' + str(round(interval * 100)), 'rb'))[s]
        env.set_para_truth(para_truth)
        budget = budget = 3.6e3
        state, observation = init_state.copy(), init_state.copy()
        done = False
        index = [0]
        score = 0
        for t in range(T):
            
            pset = env.para_sampling(100)
            para = pset[index[0]]
            action_buffer = [] 
            reward_buffer = []
            if budget > 0:     
                for j in range(num_sample):  
                    b0 = budget
                    s0 = score
                    ac = [random.random(), random.random(), random.random(), random.random(), random.random()]
                    action_buffer.append(ac) 
                    state_ex, reward, _, _, c = env.transition(state, ac, para)  
                    s0 += reward
                    b0 -= c
                    for i in range(t, T):
                        if b0 > 0:
                            act = np.ones(5) 
                            act[-1] = 0.5

                        else:
                            act = np.zeros((5))
                        next_ex, reward_ex, _, _, c = env.transition(state_ex, act, para)  
                        s0 += reward_ex
                        b0 -= c
                        state_ex = next_ex.copy()          
                    reward_buffer.append(s0)

                max_reward = np.argmax(np.array(reward_buffer))
                action = action_buffer[max_reward]
            else:
                action = np.zeros((5))

            next_state, obs, reward, cost, new_i, new_e, ck = env.online_state_estimate(state, observation, action, 1, 100, 1)
            index = ck.argsort()[0][:1]
            score += reward
            budget -= cost
            state = next_state.copy()
            observation = obs.copy()
            print(['time:', t, 'state:', state])
            print(['action:', action])
        result.append(score)
    print(result)
    print(np.mean(result))
    # print(time.time() - tic)
    
    
def online_rollout_out(interval, out):
    init_state = np.array([100000, 20, 0, 0, 0, 0])
    N = np.sum(init_state)
    n_state = 6
    n_action = 2
    T = 50
    budget = 3.6e3 
    num_sample = 20000
    deter_para = dict(N=N, B=budget, T=T, alpha=0.6, v_max=0.2 / 14,
                      cost_ti=0.0977, cost_ta=0.02, cost_v=0.07, cost_poc_0=0.000369,
                      cost_poc_1=0.001057, pid=1.10 / 1000, psr=0.7 / 3,
                      pid_plus=0.1221 / 1000, pir=1 / 8)

    in_deter_para = dict(beta2=[0.78735 * (1-interval/2), 0.78735 * (1+interval/2)],
                         beta1=[0.15747 * (1-interval/2), 0.15747 * (1+interval/2)],
                         pei=[0.10714 * (1-interval/2), 0.10714 * (1+interval/2)],
                         per=[0.04545 * (1-interval/2), 0.04545 * (1+interval/2)])

    in_deter_truth = dict(beta1=0.15747, beta2=0.78735, pei=0.10714, per=0.04545)

    para_truth = {**deter_para, **in_deter_truth}
    random.seed(0)
    tic = time.time()
    result = []
    for s in range(10):
        env = CM.Env_model(init_state, deter_para, in_deter_para, 0)
        para_truth = pickle.load(open('out_of_sample_' + str(round(interval * 100)) + '_' + str(round(out * 100)), 'rb'))[s]
        env.set_para_truth(para_truth)
        budget = budget = 3.6e3
        state, observation = init_state.copy(), init_state.copy()
        done = False
        index = [0]
        score = 0
        for t in range(T):
            pset = env.para_sampling(100)
            para = pset[index[0]]
            action_buffer = [] 
            reward_buffer = []
            if budget > 0:     
                for j in range(num_sample):  
                    b0 = budget
                    s0 = score
                    ac = [random.random(), random.random(), random.random(), random.random(), random.random()]
                    action_buffer.append(ac) 
                    state_ex, reward, _, _, c = env.transition(state, ac, para)  
                    s0 += reward
                    b0 -= c
                    for i in range(t, T):
                        if b0 > 0:
                            act = np.ones(5) 
                            act[-1] = 0.5

                        else:
                            act = np.zeros((5))
                        next_ex, reward_ex, _, _, c = env.transition(state_ex, act, para)  
                        s0 += reward_ex
                        b0 -= c
                        state_ex = next_ex.copy()          
                    reward_buffer.append(s0)

                max_reward = np.argmax(np.array(reward_buffer))
                action = action_buffer[max_reward]
            else:
                action = np.zeros((5))

            next_state, obs, reward, cost, new_i, new_e, ck = env.online_state_estimate(state, observation, action, 1, 100, 1)
            index = ck.argsort()[0][:1]
            score += reward
            budget -= cost
            state = next_state.copy()
            observation = obs.copy()
            print(['time:', t, 'state:', state])
            print(['action:', action])
        result.append(score)
    print(result)
    print(np.mean(result))
    # print(time.time() - tic)