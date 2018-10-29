# -*- coding: utf-8 -*-
import math
import numpy as np


def cal_l(data, theta):
    log_likelihood = 0
    for i in range(3, len(data)):
        log_likelihood -= 0.5*(math.log(2*math.pi) + pow(data[i]-theta[0]*data[i-1]-theta[1]*data[i-2]-theta[2]*data[i-3], 2))
    return log_likelihood


def l_gradient(data, theta):
    gradient = np.zeros((3, 1))
    for t in range(3, len(data)):
        gradient[0] += (data[t]-theta[0]*data[t-1]-theta[1]*data[t-2]-theta[2]*data[t-3])*data[t-1]
        gradient[1] += (data[t]-theta[0]*data[t-1]-theta[1]*data[t-2]-theta[2]*data[t-3])*data[t-2]
        gradient[2] += (data[t]-theta[0]*data[t-1]-theta[1]*data[t-2]-theta[2]*data[t-3])*data[t-3]
    return gradient


def l_hessian_inv(data):
    hessian = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            for t in range(3, len(data)):
                hessian[i,j] -= data[t-(i+1)]*data[t-(j+1)]
    return np.linalg.inv(hessian)


def vector_diff(the1, the2):
    diff = 0
    for i in range(len(the1)):
        diff += pow(the1[i]-the2[i], 2)
    return diff


def cal_err(data, theta):
    err = 0
    for t in range(3, len(data)):
        err += pow(data[t]-theta[0]*data[t-1]-theta[1]*data[t-2]-theta[2]*data[t-3], 2)
    return err/(len(data)-3)


def main():
    with open(r'C:\\Users\Haiya Ye\PycharmProjects\learn algorithms\markov_models\hw4_nasdaq00.txt', 'r') as f:
        indice00 = f.read().split('\n')
    indice00 = [float(b) for b in indice00]
    theta0 = np.zeros((3, 1))
    theta = np.array(theta0) - np.asarray(np.asmatrix(l_hessian_inv(indice00)) * np.asmatrix(l_gradient(indice00, theta0)))
    times = 0
    while vector_diff(theta0, theta) > 0.001:
        times += 1
        theta0 = theta
        theta = np.array(theta0) - np.asarray(np.asmatrix(l_hessian_inv(indice00))*np.asmatrix(l_gradient(indice00, theta0)))
    print(theta)
    with open(r'C:\\Users\Haiya Ye\PycharmProjects\learn algorithms\markov_models\hw4_nasdaq01.txt', 'r') as f:
        indice01 = f.read().split('\n')
    indice01 = [float(b) for b in indice01]
    print('mean squared error on the data from 2000 is', cal_err(indice00, theta))
    print('mean squared error on the data from 2001 is', cal_err(indice01, theta))


if __name__ == "__main__":
    main()