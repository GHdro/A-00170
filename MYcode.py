# -*- coding: utf-8 -*-
import numpy as np
from numpy import random
import scipy.io as scio
from math import sqrt
from sklearn import preprocessing
from scipy import linalg as LA
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import datetime
import h5py

ALPHA_PRIME = 0.5
BETA_PRIME = 0.5

KAPPA_MAX_PRIME = 20000.0
KAPPA_MIN_PRIME = 5000.0

DELTA = 0.5

EPSILON_1 = 1e-8
EPSILON_2 = 1e-8

def classification_accuracy(predict_label, Label):
    count = 0
    label = Label.argmax(axis=1)
    prediction = predict_label.argmax(axis=1)
    for j in list(range(Label.shape[0])):
        if label[j] == prediction[j]:
            count += 1
    return round(count / len(Label), 8)

def tansig(x):
    return (2 / (1 + np.exp(-2 * x))) - 1

def relu(data):
    return np.maximum(data, 0)

def pseudo_inv(A, reg):
    A_p = np.linalg.pinv(A.T.dot(A)).dot(A.T)
    return np.array(A_p)

def shrinkage(a, b):
    z = np.maximum(a - b, 0) - np.maximum(-a - b, 0)
    return z

def sparse_weights(A, b):
    lam = 0.001
    itrs = 50
    AA = A.T.dot(A)
    m = A.shape[1]
    n = b.shape[1]
    x1 = np.zeros([m, n])
    wk = x1
    ok = x1
    uk = x1
    L1 = np.mat(AA + np.eye(m)).I
    L2 = (L1.dot(A.T)).dot(b)
    for i in range(itrs):
        ck = L2 + np.dot(L1, (ok - uk))
        ok = shrinkage(ck + uk, lam)
        uk = uk + ck - ok
        wk = ok
    return wk

def preprocess(train_x):
    N1 = 10
    N2 = 10
    N3 = 670
    s = 300
    train_x = preprocessing.scale(train_x, axis=1)
    train_x_bias = np.hstack([train_x, 0.1 * np.ones((train_x.shape[0], 1))])
    Z = np.zeros([train_x.shape[0], N2 * N1])
    We_set = []
    max_min = []
    min_value = []
    for i in range(N2):
        random.seed(i)
        We = 2 * random.randn(train_x.shape[1] + 1, N1) - 1
        X_We_B = np.dot(train_x_bias, We)
        scaler1 = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(X_We_B)
        feature1 = scaler1.transform(X_We_B)
        We_star = sparse_weights(feature1, train_x_bias).T
        We_set.append(We_star)
        Zi = np.dot(train_x_bias, We_star)
        max_min.append(np.max(Zi, axis=0) - np.min(Zi, axis=0))
        min_value.append(np.min(Zi, axis=0))
        Zi = (Zi - min_value[i]) / max_min[i]
        Z[:, N1 * i:N1 * (i + 1)] = Zi
        del Zi, X_We_B, We
    Z_bias = np.hstack([Z, 0.1 * np.ones((Z.shape[0], 1))])
    if N1 * N2 >= N3:
        random.seed(67797325)
        Wh = LA.orth(2 * random.randn(N2 * N1 + 1, N3)) - 1
    else:
        random.seed(67797325)
        Wh = LA.orth(2 * random.randn(N2 * N1 + 1, N3).T - 1).T
    Z_Wh_B = np.dot(Z_bias, Wh)
    param_shrink = s / np.max(Z_Wh_B)
    H = tansig(Z_Wh_B * param_shrink)
    A = np.hstack([Z, H])
    return Z, H, A, N1, N2, N3

def log_liklihoods(OutputWeight, A, train_y):
    OutputWeight = torch.from_numpy(OutputWeight)
    OutputWeight.requires_grad = True
    A = torch.from_numpy(A)
    output = torch.mm(A, OutputWeight)
    label = torch.from_numpy(train_y)
    log_liklihoods = F.cross_entropy(output, label.max(1)[1])
    log_liklihoods.backward()
    FIM = OutputWeight.grad ** 2
    return FIM.numpy() * train_y.shape[0]

def confusion_matrix(baseline, result):
    max_len = max((len(l) for l in result))
    result = list(map(lambda l: l + [0] * (max_len - len(l)), result))
    result = torch.Tensor(result)
    baseline = torch.Tensor(baseline)
    nt = result.size(0)
    acc = result.diag()
    fin = result[nt - 1]
    bwt = fin - acc
    fwt = acc - baseline
    return fin.mean(), bwt[0:(nt - 1)].mean(), fwt[1:nt].mean()

def calculate_cosine_similarity(a, b, eps=1e-8):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm < eps or b_norm < eps:
        return 0.0
    return np.dot(a, b) / (a_norm * b_norm)

def calculate_task_similarity(task_i_info, task_j_info):
    avg_activation_i = task_i_info['avg_activation']
    avg_activation_j = task_j_info['avg_activation']
    sim_ij = calculate_cosine_similarity(avg_activation_i, avg_activation_j, EPSILON_1)
    cov_i = task_i_info['covariance']
    cov_j = task_j_info['covariance']
    frob_diff = np.linalg.norm(cov_i - cov_j, 'fro')
    frob_sum = np.linalg.norm(cov_i, 'fro') + np.linalg.norm(cov_j, 'fro')
    r_ij = frob_diff / (frob_sum + EPSILON_2)
    IS_ij = ALPHA_PRIME * sim_ij + BETA_PRIME * r_ij
    IS_ij = np.clip(IS_ij, 0.0, 1.0)
    return IS_ij

def calculate_kappa(IS_ij, is_early_stage):
    tau_ij = 2 * IS_ij - 1
    if is_early_stage:
        kappa_min = KAPPA_MIN_PRIME
        kappa_max = KAPPA_MAX_PRIME * 0.5
    else:
        kappa_min = KAPPA_MIN_PRIME * 2
        kappa_max = KAPPA_MAX_PRIME
    lambda_param = 0.25 if is_early_stage else 0.4
    exp_2tau = np.exp(2 * tau_ij)
    exp_neg2tau = np.exp(-2 * tau_ij)
    theta_ij = (kappa_max - kappa_min) / (2 * exp_2tau + 2 * exp_neg2tau)
    if tau_ij == 1:
        kappa = kappa_max
    elif tau_ij == -1:
        kappa = kappa_min
    elif tau_ij >= 0:
        kappa = kappa_min + theta_ij * ((2 - lambda_param) * exp_2tau + lambda_param * exp_neg2tau)
    else:
        kappa = kappa_min + theta_ij * ((1 + lambda_param) * exp_2tau + (1 - lambda_param) * exp_neg2tau)
    return kappa

def calculate_adaptive_lambdas(current_task_idx, learned_tasks_info, total_tasks):
    if current_task_idx == 0:
        return []
    lambdas = []
    is_early_stage = (current_task_idx + 1) <= int(DELTA * total_tasks)
    for i, task_info in enumerate(learned_tasks_info):
        IS_ij = calculate_task_similarity(learned_tasks_info[-1], task_info)
        kappa_ij = calculate_kappa(IS_ij, is_early_stage)
        lambdas.append(kappa_ij)
    return np.array(lambdas)

def TOFIMC(A, train_y, chain_task_index, FIM, lamb):
    L = A.shape[1]
    STsumlF = np.zeros((train_y.shape[1], L, L))
    for q in np.arange(train_y.shape[1]):
        sum_lambF = np.zeros((L, L))
        for t in range(1, chain_task_index):
            FIM_q = np.diag(FIM[t - 1][:, q])
            sum_lambF += lamb[t - 1] * FIM_q
        STsumlF[q, :, :] = sum_lambF
    return STsumlF

def train(train_x, train_y, dl, C, Alpha, Lmax, chain_task_index, FIM, lamb, OutputWeight_ed, sample_counts, prev_A):
    time_start = time.time()
    Z, H, A, N1, N2, N3 = preprocess(train_x)
    Zmax = np.max(Z)
    train_x = A
    np.random.seed(2505)
    InputWeight = Alpha * (2 * np.random.rand(Lmax, train_x.shape[1]) - 1)
    InputBias = Alpha * (2 * np.random.rand(Lmax, 1) - 1)
    tempH = np.dot(InputWeight, train_x.T) + InputBias
    H = relu(tempH.T)
    A = np.hstack([train_x, H]) if dl == 1 else H
    ATA = A.T.dot(A)
    eig_values = np.linalg.eigvalsh(ATA)
    min_eig_value = np.min(eig_values)
    Qt = None
    L = A.shape[1]
    STsumlF = np.zeros((train_y.shape[1], L, L))

    if chain_task_index == 1:
        A_p = pseudo_inv(A, C)
        OutputWeight = np.dot(A_p, train_y)
    else:
        OutputWeight = np.empty((L, 0))
        for q in np.arange(train_y.shape[1]):
            sum_lambF = np.zeros((L, L))
            for t in range(1, chain_task_index):
                FIM_q = np.diag(FIM[t - 1][:, q])
                sum_lambF += lamb[t - 1] * FIM_q
            STsumlF[q, :, :] = sum_lambF
            beta_q = pseudo_inv(ATA + sum_lambF + 1e-5 * np.eye(L), C).dot(
                A.T.dot(train_y[:, q:q + 1]) + sum_lambF.dot(OutputWeight_ed[:, q:q + 1]))
            OutputWeight = np.concatenate((OutputWeight, beta_q), axis=1)

    if chain_task_index > 1 and OutputWeight_ed is not None and prev_A is not None:
        deltW = OutputWeight - OutputWeight_ed
        t_er = np.dot(prev_A, deltW)
        Qt = np.sum(np.square(t_er)) / sample_counts[chain_task_index - 2]

    train_output = np.dot(A, OutputWeight)
    train_acc = classification_accuracy(train_output, train_y)
    time_end = time.time()
    return InputWeight, InputBias, OutputWeight, A, Qt, STsumlF

def last_train(train_x, train_y, dl, C, Alpha, Lmax, ORset):
    time_start = time.time()
    Z, H, A, N1, N2, N3 = preprocess(train_x)
    Zmax = np.max(Z)
    train_x = A
    np.random.seed(2505)
    InputWeight = Alpha * (2 * np.random.rand(Lmax, train_x.shape[1]) - 1)
    InputBias = Alpha * (2 * np.random.rand(Lmax, 1) - 1)
    tempH = np.dot(InputWeight, train_x.T) + InputBias
    H = relu(tempH.T)
    A = np.hstack([train_x, H]) if dl == 1 else H
    ATA = A.T.dot(A)
    eig_values = np.linalg.eigvalsh(ATA)
    min_eig_value = np.min(eig_values)
    L = A.shape[1]
    OutputWeight = np.empty((L, 0))
    IIF = [ow for chain in ORset for ow in chain['sum_labmf']]
    OWd = [ow for chain in ORset for ow in chain['OutputWeights']]
    for q in np.arange(train_y.shape[1]):
        sum_lambF = np.zeros((L, L))
        sbF = np.zeros((L, 1))
        sbQ = np.zeros((L, L))
        for t in range(len(ORset)):
            STsumlF = np.array(IIF[t][0])
            OutputWeight_ed = np.array(OWd[t][0])
            sum_lambF = STsumlF[q][:][:]
            sbF += sum_lambF.dot(OutputWeight_ed[:, q:q + 1])
            sbQ += sum_lambF
        beta_q = pseudo_inv(ATA + sbQ + 1e-5 * np.eye(L), C).dot(A.T.dot(train_y[:, q:q + 1]) + sbF)
        OutputWeight = np.concatenate((OutputWeight, beta_q), axis=1)
    train_output = np.dot(A, OutputWeight)
    train_acc = classification_accuracy(train_output, train_y)
    time_end = time.time()
    return InputWeight, InputBias, OutputWeight, A

def test(test_x, test_y, dl, InputWeight, InputBias, OutputWeight, task_index):
    time_start = time.time()
    Z, H, A, N1, N2, N3 = preprocess(test_x)
    test_x = A
    tempH_test = np.dot(InputWeight, test_x.T) + InputBias
    H_test = relu(tempH_test.T)
    A_test = np.hstack([test_x, H_test]) if dl == 1 else H_test
    test_output = np.dot(A_test, OutputWeight)
    test_acc = classification_accuracy(test_output, test_y)
    time_end = time.time()
    test_time = time_end - time_start
    return test_acc

def main(task_orders, TQ):
    dataFile = 'SVHN_split5_CIL.mat'
    data = scio.loadmat(dataFile)
    dl = 0
    ar = 0.5
    Alpha = 1
    Lmax = 900
    C = 2 ** -30
    FIM = []
    TN = TQ
    R = []
    Ri = []
    sample_counts = []
    chains = []
    ORset = []
    OW = []
    G = []
    PreI = []
    learned_tasks_info = []

    time_start = time.time()
    first_task = task_orders[0]
    train_x = np.double(data[f'train_x_{first_task}'])
    train_y = np.double(data[f'train_y_{first_task}'])
    sample_counts.append(train_x.shape[0])
    Z, H, A_for_info, _, _, _ = preprocess(train_x)
    avg_activation = np.mean(A_for_info, axis=0)
    covariance = np.cov(train_x.T)
    IW, IB, OW, G, _, _ = train(train_x, train_y, dl, C, Alpha, Lmax, 1, FIM, [], None, sample_counts, None)
    fim = log_liklihoods(OW, G, train_y)
    learned_tasks_info.append({
        'task_id': first_task,
        'avg_activation': avg_activation,
        'covariance': covariance,
        'FIM': fim,
        'OutputWeight': OW.copy(),
        'A_matrix': G.copy()
    })
    adaptive_lamb = calculate_adaptive_lambdas(0, learned_tasks_info, TQ)
    chains.append({'tasks': [first_task]})
    FIM.append(fim)
    STsumlF = np.zeros((train_y.shape[1], G.shape[1], G.shape[1]))
    for q in np.arange(train_y.shape[1]):
        FIM_q = np.diag(FIM[0][:, q])
        STsumlF[q, :, :] = 6000 * FIM_q
    test_x = np.double(data[f'test_x_{first_task}'])
    test_y = np.double(data[f'test_y_{first_task}'])
    test_accij = test(test_x, test_y, dl, IW, IB, OW, first_task)
    Ri.append(test_accij)
    R.append(Ri)
    Ri = []
    PreI.append({'OutputWeights': [OW], 'sum_labmf': [STsumlF]})

    for global_idx in range(1, TN - 1):
        current_task = task_orders[global_idx]
        train_x = np.double(data[f'train_x_{current_task}'])
        train_y = np.double(data[f'train_y_{current_task}'])
        sample_counts.append(train_x.shape[0])
        Z, H, A_for_info, _, _, _ = preprocess(train_x)
        current_avg_activation = np.mean(A_for_info, axis=0)
        current_covariance = np.cov(train_x.T)
        learned_tasks_info.append({
            'task_id': current_task,
            'avg_activation': current_avg_activation,
            'covariance': current_covariance,
            'FIM': None,
            'OutputWeight': None,
            'A_matrix': None
        })
        adaptive_lamb = calculate_adaptive_lambdas(global_idx, learned_tasks_info, TQ)
        learned_tasks_info.pop()
        last_chain = chains[-1]
        chain_task_idx = len(last_chain['tasks']) + 1
        IW, IB, OW, G, Qt, STsumlF = train(train_x, train_y, dl, C, Alpha, Lmax, chain_task_idx, FIM, adaptive_lamb, OW,
                                           sample_counts, G)
        if Qt is not None and Qt > ar:
            OW_hist = PreI[0]['OutputWeights']
            STsumlF_hist = PreI[0]['sum_labmf']
            ORset.append({'OutputWeights': [OW_hist], 'sum_labmf': [STsumlF_hist]})
            PreI = []
            OW = []
            G = []
            FIM = []
            IW, IB, OW, G, _, _ = train(train_x, train_y, dl, C, Alpha, Lmax, 1, FIM, [], None, sample_counts, None)
            fim = log_liklihoods(OW, G, train_y)
            FIM.append(fim)
            STsumlF = np.zeros((train_y.shape[1], G.shape[1], G.shape[1]))
            for q in np.arange(train_y.shape[1]):
                FIM_q = np.diag(fim[:, q])
                STsumlF[q, :, :] = adaptive_lamb[global_idx] * FIM_q if len(adaptive_lamb) > 0 else 0
            learned_tasks_info.append({
                'task_id': current_task,
                'avg_activation': current_avg_activation,
                'covariance': current_covariance,
                'FIM': fim,
                'OutputWeight': OW.copy(),
                'A_matrix': G.copy()
            })
            chains.append({'tasks': [current_task]})
            PreI.append({'OutputWeights': [OW], 'sum_labmf': [STsumlF]})
        else:
            PreI = []
            last_chain['tasks'].append(current_task)
            fim = log_liklihoods(OW, G, train_y)
            FIM.append(fim)
            learned_tasks_info.append({
                'task_id': current_task,
                'avg_activation': current_avg_activation,
                'covariance': current_covariance,
                'FIM': fim,
                'OutputWeight': OW.copy(),
                'A_matrix': G.copy()
            })
            current_next_task = task_orders[global_idx + 1]
            next_train_y = np.double(data[f'train_y_{current_next_task}'])
            STsumlF = TOFIMC(G, next_train_y, chain_task_idx + 1, FIM, adaptive_lamb)
            PreI.append({'OutputWeights': [OW], 'sum_labmf': [STsumlF]})

        for j in task_orders[:global_idx + 1]:
            test_x = np.double(data[f'test_x_{j}'])
            test_y = np.double(data[f'test_y_{j}'])
            test_accij = test(test_x, test_y, dl, IW, IB, OW, j)
            Ri.append(test_accij)
        R.append(Ri)
        Ri = []
    final_task = task_orders[-1]
    train_x = np.double(data[f'train_x_{final_task}'])
    train_y = np.double(data[f'train_y_{final_task}'])

    if len(chains) > 1:
        OW_final = PreI[0]['OutputWeights']
        STsumlF_final = PreI[0]['sum_labmf']
        ORset.append({'OutputWeights': [OW_final], 'sum_labmf': [STsumlF_final]})
        final_IW, final_IB, final_OW, final_G = last_train(train_x, train_y, dl, C, Alpha, Lmax, ORset)
    else:
        last_chain['tasks'].append(final_task)
        chain_task_idx = len(task_orders)
        sample_counts.append(train_x.shape[0])
        final_IW, final_IB, final_OW, final_G, Qt, STsumlF = train(train_x, train_y, dl, C, Alpha, Lmax, chain_task_idx,
                                                                   FIM, adaptive_lamb, OW, sample_counts, G)

    final_R = []
    for j in task_orders:
        test_x = np.double(data[f'test_x_{j}'])
        test_y = np.double(data[f'test_y_{j}'])
        test_acc = test(test_x, test_y, dl, final_IW, final_IB, final_OW, j)
        final_R.append(test_acc)
    R.append(final_R)

    baseline = []
    for i in task_orders:
        train_x = np.double(data[f'train_x_{i}'])
        train_y = np.double(data[f'train_y_{i}'])
        test_x = np.double(data[f'test_x_{i}'])
        test_y = np.double(data[f'test_y_{i}'])
        IW, IB, OW, G, _, _ = train(train_x, train_y, dl, C, Alpha, Lmax, 1, [], [], None, sample_counts, None)
        test_acc = test(test_x, test_y, dl, IW, IB, OW, i)
        baseline.append(test_acc)
    acc, bwt, fwt = confusion_matrix(baseline, R)
    max_len = max((len(l) for l in R))
    R = list(map(lambda l: l + [0] * (max_len - len(l)), R))
    diagonal = [R[i][i] for i in range(len(R))]
    time_end = time.time()
    all_performed_time = time_end - time_start
    return acc, bwt, fwt, all_performed_time, diagonal

if __name__ == '__main__':
    Multiple = 5
    ACC = []
    BWT = []
    FWT = []
    Time = []
    AQ = []
    TQ = 5
    with h5py.File("TO.h5", 'r') as f:
        tosset = f['Taskorder'][:]
    tosset = np.asarray(tosset)
    for multi_runs in range(Multiple):
        tos = tosset[multi_runs, :]
        acc, bwt, fwt, all_performed_time, diagonal = main(tos, TQ)
        ACC.append(acc)
        BWT.append(bwt)
        FWT.append(fwt)
        Time.append(all_performed_time)
        AQ.append(diagonal)
    print('Results of {} repeated runs'.format(Multiple))
    print('ACC: mean {:.4f}, std {:.4f}'.format(np.mean(ACC), np.std(ACC, ddof=1)))
    print('Time: mean {:.4f}, std {:.4f}'.format(np.mean(Time), np.std(Time, ddof=1)))