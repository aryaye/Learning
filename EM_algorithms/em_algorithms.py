# -*- coding: utf-8 -*-
import math


def pzxxy(x, y, p, i):
    if x[i] == 0 or y == 0:
        return 0
    multi = 1
    for idx, p_j in enumerate(p):
        if x[idx] == 1:
            multi *= (1-p_j)
    return y*x[i]*p[i]/(1-multi)


def update(xdata, ydata, p):
    new_p = []
    for i, p_i, in enumerate(p):
        sums = 0
        ti = 0
        for idx, x in enumerate(xdata):
            sums += pzxxy(x, ydata[idx], p, i)
            ti += x[i]
        new_p.append(sums/ti)
    return new_p


def pyx(y, x, p):
    multi = 1
    for i, xi in enumerate(x):
        if xi == 1:
            multi *= (1-p[i])
    if y == 1:
        return 1-multi
    else:
        return multi


def log_likelihood(xdata, ydata, p):
    ans = 0
    for idx, x in enumerate(xdata):
        ans += math.log(pyx(ydata[idx], x, p))
    return ans/(idx+1)


def mistake(xdata, ydata, p):
    count = 0
    for idx, x in enumerate(xdata):
        if ydata[idx] == 0:
            if pyx(1, x, p) >= 0.5:
                count += 1
        else:
            if pyx(1, x, p) <= 0.5:
                count += 1
    return count


def main():
    # processing data
    with open(r'C:\\Users\Haiya Ye\PycharmProjects\learn algorithms\EM_algorithms\spectX.txt', 'r') as f:
        xdata = f.read().split('\n')[:-1]
    xdata = [[int(xi) for xi in x.split(' ') if xi != ''] for x in xdata]
    with open(r'C:\\Users\Haiya Ye\PycharmProjects\learn algorithms\EM_algorithms\spectY.txt', 'r') as f:
        ydata = f.read().split('\n')[:-1]
    ydata = [int(y) for y in ydata]
    dimension = len(xdata[0])
    p = [1/dimension]*dimension
    iteration = 0
    prints = [0,1,2,4,8,16,32,64,128,256]
    while iteration < 257:
        if iteration  in prints:
            print(mistake(xdata, ydata, p))
            print(log_likelihood(xdata, ydata, p))
        p = update(xdata, ydata, p)
        iteration += 1


if __name__ == '__main__':
    main()