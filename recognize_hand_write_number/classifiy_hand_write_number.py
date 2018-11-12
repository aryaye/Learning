# -*- coding: utf-8 -*-
import numpy
from math import exp
from math import log
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1.0 / (1 + exp(-x))


def inner(x, theta):
    return sum([x[i]*theta[0, i] for i in range(len(x))])


def f_likelihood(theta, data3, data5):
    loglikelihood = 0
    # label 3 indicate 0 in regular regression
    for i in range(len(data3)):
        logit = inner(data3[i], theta)
        loglikelihood -= log(1 + exp(-logit))
        loglikelihood -= logit
    # label 5 indicate 1 in regular regression
    for i in range(len(data5)):
        logit = inner(data5[i], theta)
        loglikelihood -= log(1 + exp(-logit))
    return loglikelihood


def f_gradient(theta, data3, data5):
    length = 64
    dl = numpy.mat(numpy.zeros((1, 64)))
    for i in range(len(data3)):
        logit = inner(data3[i], theta)
        for k in range(length):
            dl[0, k] += data3[i][k] * (1 - sigmoid(logit))
            dl[0, k] -= data3[i][k]
    for i in range(len(data5)):
        logit = inner(data5[i], theta)
        for k in range(length):
            dl[0, k] += data5[i][k] * (1 - sigmoid(logit))
    return dl


def f_hessian(theta, data3, data5):
    length = 64
    h = numpy.mat(numpy.zeros((length, length)))
    for t in range(len(data3)):
        logit = inner(data3[t], theta)
        for i in range(length):
            for j in range(i, length):
                h[i, j] -= data3[t][i]*data3[t][j]*exp(-logit)*pow(sigmoid(logit), 2)
                h[j, i] = h[i, j]
    for t in range(len(data5)):
        logit = inner(data5[t], theta)
        for i in range(length):
            for j in range(i, length):
                h[i, j] -= data5[t][i]*data5[t][j]*exp(-logit)*pow(sigmoid(logit), 2)
                h[j, i] = h[i, j]
    return h.I


def err(m1, m2):
    return (m1-m2)*(m1-m2).T


def err_r(theta, data3, data5):
    fault = 0
    for i in data3:
        if inner(i, theta) > 0.5:
            fault += 1
    for j in data5:
        if inner(j, theta) < 0.5:
            fault += 1
    return fault/(len(data5)+len(data3))

def main():
    with open(r'C:\\Users\Haiya Ye\PycharmProjects\learn algorithms\recognize_hand_write_number\new_train3.txt', 'r') as f:
        lines = f.read().split('\n')
    data3 = []
    for line in lines[:len(lines)-1]:
        m = []
        for i in line.split(' ')[:64]:
            m.append(int(i))
        data3.append(m)
    with open(r'C:\\Users\Haiya Ye\PycharmProjects\learn algorithms\recognize_hand_write_number\new_train5.txt', 'r') as f:
        lines = f.read().split('\n')
    data5 = []
    for line in lines[:len(lines)-1]:
        m = []
        for i in line.split(' ')[:64]:
            m.append(int(i))
        data5.append(m)
    # question 1
    y = []
    err_rate = []
    theta0 = numpy.mat(numpy.zeros((1, 64)))
    y.append(f_likelihood(theta0, data3, data5))
    err_rate.append(err_r(theta0, data3, data5))
    theta1 = theta0 - f_gradient(theta0, data3, data5)*f_hessian(theta0, data3, data5)
    y.append(f_likelihood(theta1, data3, data5))
    err_rate.append(err_r(theta1, data3, data5))
    err_limit = 0.01
    while err(theta0, theta1) > err_limit:
        theta0 = theta1
        theta1 = theta0 - f_gradient(theta0, data3, data5)*f_hessian(theta0, data3, data5)
        y.append(f_likelihood(theta1, data3, data5))
        err_rate.append(err_r(theta1, data3, data5))
        print(err(theta1, theta0))
    print(y)       # log-likelihood each iteration
    plt.plot(range(len(y)), y)
    print(err_rate)  # percent err rate each iteration
    plt.plotrange(range(len(err_rate)),err_rate)
    print(theta1)
    # question2
    with open(r'C:\\Users\Haiya Ye\PycharmProjects\learn algorithms\recognize_hand_write_number\new_test3.txt', 'r') as f:
        lines = f.read().split('\n')
    test3 = []
    for line in lines[:len(lines)-1]:
        m = []
        for i in line.split(' ')[:64]:
            m.append(int(i))
        test3.append(m)
    with open(r'C:\\Users\Haiya Ye\PycharmProjects\learn algorithms\recognize_hand_write_number\new_test5.txt', 'r') as f:
        lines = f.read().split('\n')
    test5 = []
    for line in lines[:len(lines)-1]:
        m = []
        for i in line.split(' ')[:64]:
            m.append(int(i))
        test5.append(m)
    print(err_r(theta1, test3, test5))



if __name__ == "__main__":
    main()