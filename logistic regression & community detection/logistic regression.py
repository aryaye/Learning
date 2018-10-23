# -*- coding: utf-8 -*-
import numpy
from urllib.request import urlopen
import scipy.optimize
import random
from math import exp
from math import log
import matplotlib.pyplot as plt


def parseData(fname):
  for l in urlopen(fname):
    yield eval(l)


def feature(datum):
    feat = [1, datum['review/taste'], datum['review/appearance'], datum['review/aroma'], datum['review/palate'], datum['review/overall']]
    return feat


def sigmoid(x):
    return 1.0 / (1 + exp(-x))


def inner(x,y):
    return sum([x[i]*y[i] for i in range(len(x))])


def f(theta, X, y, lam):
    loglikelihood = 0
    for i in range(len(X)):
        logit = inner(X[i], theta)
        loglikelihood -= log(1 + exp(-logit))
        if not y[i]:
            loglikelihood -= logit
    for k in range(len(theta)):
        loglikelihood -= lam * theta[k]*theta[k]
    return -loglikelihood


def fprime(theta, X, y, lam):
    dl = [0]*len(theta)
    for i in range(len(X)):
        logit = inner(X[i], theta)
        for k in range(len(theta)):
            dl[k] += X[i][k] * (1 - sigmoid(logit))
            if not y[i]:
                dl[k] -= X[i][k]
    for k in range(len(theta)):
        dl[k] -= lam*2*theta[k]
    return numpy.array([-x for x in dl])


def train(lam):
    theta,_,_ = scipy.optimize.fmin_l_bfgs_b(f, [0]*len(X_train[0]), fprime, pgtol = 10, args = (X_train, y_train, lam))
    return theta


def performance(theta, X, y):
    scores = [inner(theta, x) for x in X]
    predictions = [s > 0 for s in scores]
    correct = [(a == b) for (a, b) in zip(predictions, y)]
    acc = sum(correct) * 1.0 / len(correct)
    return acc


def acc(theta, X_test, y_test):
    test_scores = [inner(theta, x) for x in X_test]

    # positive
    print("positive: " + str(sum(y_test)))

    # negative
    print("negative: " + str(len(y_test) - sum(y_test)))

    tp = 0  # true positive
    tn = 0  # true negative
    fp = 0  # false positive
    fn = 0  # false negative
    for i in range(len(test_scores)):
        if test_scores[i] > 0 and y_test[i]:
            tp += 1
        if test_scores[i] < 0:
            if not y_test[i]:
                tn += 1
        if test_scores[i] > 0:
            if not y_test[i]:
                fp += 1
        if test_scores[i] < 0 and y_test[i]:
            fn += 1

    print("true positive: " + str(tp))
    print("true negative: " + str(tn))
    print("false positive: " + str(fp))
    print("false negative: " + str(fn))


def f_m(theta, X, y, lam):
    loglikelihood = 0
    for i in range(len(X)):
        logit = inner(X[i], theta)
        if y[i]:
            loglikelihood -= log(1 + exp(-logit))
        if not y[i]:
            loglikelihood -= 10 * logit
            loglikelihood -= 10 * log(1 + exp(-logit))
    for k in range(len(theta)):
        loglikelihood -= lam * theta[k]*theta[k]
    return -loglikelihood


# NEGATIVE Derivative of log_lp_likelihood
def fprime_m(theta, X, y, lam):
    dl = [0]*len(theta)
    for i in range(len(X)):
        logit = inner(X[i], theta)
        for k in range(len(theta)):
            if y[i]:
                dl[k] += X[i][k] * (1 - sigmoid(logit))
            if not y[i]:
                dl[k] -= 10 * X[i][k]
                dl[k] += 10 * X[i][k] * (1 - sigmoid(logit))
    for k in range(len(theta)):
        dl[k] -= lam*2*theta[k]
    return numpy.array([-x for x in dl])


def train_m(lam):
    theta,_,_ = scipy.optimize.fmin_l_bfgs_b(f_m, [0]*len(X_train[0]), fprime_m, pgtol = 10, args = (X_train, y_train, lam))
    return theta


if __name__ == "__main__":
    # preprocessing the data
    print("Reading data...")
    data = list(parseData("http://jmcauley.ucsd.edu/cse190/data/beer/beer_50000.json"))
    print("done")
    print("spliting....")
    random.shuffle(data)
    train_data = data[:16667]
    validation = data[16667:33334]
    test_data = data[33334:]
    print("done")

    # train data with the classic logistic regression methods
    X_train = [feature(d) for d in train_data]
    y_train = [d['beer/ABV'] >= 6.5 for d in train_data]
    theta = train(1.0)

    # the accuracy
    X_test = [feature(d) for d in test_data]
    y_test = [d['beer/ABV'] >= 6.5 for d in test_data]
    X_validation = [feature(d) for d in validation]
    y_validation = [d['beer/ABV'] >= 6.5 for d in validation]
    print('the accuracy of test set is', performance(theta, X_test, y_test))
    print('the accuracy of validation set is', performance(theta, X_validation, y_validation))

    acc(theta, X_test, y_test)
    print(theta)

    # train the data with the importance clarity
    theta_m = train_m(1.0)
    print(theta_m)
    acc(theta_m, X_test, y_test)

    # implement pipeline to select best model based on performance on the validation set
    lama = [0, 0.01, 0.1, 1, 100]
    performances = []
    X_validation = [feature(d) for d in validation]
    y_validation = [d['beer/ABV'] >= 6.5 for d in validation]

    for lamada in lama:
        theta = train(lamada)
        performances.append(performance(theta, X_validation, y_validation))

    plt.figure()
    plt.plot(lama, performances)
