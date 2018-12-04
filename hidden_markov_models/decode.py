# -*- coding: utf-8 -*-
from math import *


def most_likely_state(emission, transition, observations, initial):
    record = []
    T = len(observations)
    J = len(initial)
    for t in range(T):
        record.append([-1]*J)
    base = [log(initial[idx])+log(emission[idx][observations[0]]) for idx in range(J)]
    for t in range(1, T):
        forward = []
        for j in range(J):
            temp = [base[i] + log(transition[i][j]) + log(emission[j][observations[t]]) for i in range(J)]
            record[t][j] = temp.index(max(temp))
            forward.append(max(temp))
        base = forward
    sequence = [base.index(max(base))]
    for t in range(T-2,-1,-1):
        sequence.append(record[t+1][sequence[-1]])
    return sequence


def decode(sequence):
    pre = sequence[0]
    ans = [chr(pre + ord('a'))]
    for string in sequence[1:]:
        if string != pre:
            if string == 26:
                ans.append(' ')
            else:
                ans.append(chr(string + ord('a')))
            pre = string
    return ''.join(ans[::-1])


def main():
    with open(r'C:\\Users\Haiya Ye\PycharmProjects\learn algorithms\hidden_markov_models\observations.txt', 'r') as f:
        observations = f.read().split(' ')[:-1]
    observations = [int(obs) for obs in observations]
    with open(r'C:\\Users\Haiya Ye\PycharmProjects\learn algorithms\hidden_markov_models\transitionMatrix.txt', 'r') as f:
        transition = f.read().split('\n')[:-1]
    transition = [[float(x) for x in tran] for tran in transition]
    transition = [data.split(' ')[:-1] for data in transition]
    with open(r'C:\\Users\Haiya Ye\PycharmProjects\learn algorithms\hidden_markov_models\emissionMatrix.txt', 'r') as f:
        emission = f.read().split('\n')[:-1]
    emission = [data.split('\t') for data in emission]
    emission = [[float(x) for x in emi] for emi in emission]
    with open(r'C:\\Users\Haiya Ye\PycharmProjects\learn algorithms\hidden_markov_models\initialStateDistribution.txt', 'r') as f:
        initial = f.read().split('\n')[:-1]
    initial = [float(x) for x in initial]
    sequence = most_likely_state(emission, transition, observations, initial)
    print(decode(sequence))


if __name__ == '__main__':
    main()