# -*- coding: utf-8 -*-


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