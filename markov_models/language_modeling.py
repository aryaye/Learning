# -*- coding: utf-8 -*-
import numpy as np
import math
import matplotlib.pyplot as plt


def sentence_probability(sentence, uniprob, biprob, word_list):
    sentence = sentence.upper().split(' ')
    p1 = 1
    p2 = 1
    for word in sentence:
        p1 *= uniprob[word_list.index(word)]
    for i in range(len(sentence)):
        if i == 0:
            p2 *= biprob[word_list.index('<s>'), word_list.index(sentence[i])]
        else:
            if biprob[word_list.index(sentence[i-1]), word_list.index(sentence[i])] == 0:
                print('the word pattern', sentence[i-1], sentence[i], 'is not observed in training data')
            p2 *= biprob[word_list.index(sentence[i-1]), word_list.index(sentence[i])]
    if p2 == 0:
        return [math.log(p1), -float('inf')]
    return [math.log(p1), math.log(p2)]


def sentence_probability_mix(sentence, uniprob, biprob, word_list, lamda):
    sentence = sentence.upper().split(' ')
    p = 1
    for i in range(len(sentence)):
        p1 = uniprob[word_list.index(sentence[i])]
        if i == 0:
            p2 = biprob[word_list.index('<s>'), word_list.index(sentence[i])]
        else:
            p2 = biprob[word_list.index(sentence[i-1]), word_list.index(sentence[i])]
        p *= ((1-lamda)*p1 + lamda*p2)
    if p == 0:
        return -float('inf')
    return math.log(p)


def main():
    # question1
    a_start = []
    with open(r'C:\\Users\Haiya Ye\PycharmProjects\learn algorithms\markov_models\hw4_unigram.txt', 'r') as f:
        count1 = f.read().split(sep='\n')[:500]
    with open(r'C:\\Users\Haiya Ye\PycharmProjects\learn algorithms\markov_models\hw4_vocab.txt', 'r') as f:
        words = f.read().split(sep='\n')[:500]
    count1 = [int(b) for b in count1]
    probability1 = [b / sum(count1) for b in count1]
    for i in range(len(count1)):
        if words[i][0] == 'A':
            a_start.append([words[i], probability1[i]])
    print('the probability of words started with A')
    for b in a_start:
        print(b)
    # question2
    count_table = np.zeros((500,500))
    probability2 = np.zeros((500,500))
    with open(r'C:\\Users\Haiya Ye\PycharmProjects\learn algorithms\markov_models\hw4_bigram.txt', 'r') as f:
        count2 = f.read().split(sep='\n')[:144981]
    count2 = [b.split('\t') for b in count2]
    for count_pattern in count2:
        i = int(count_pattern[0])
        j = int(count_pattern[1])
        count_table[i-1, j-1] = int(count_pattern[2])
        probability2[i-1,j-1] = int(count_pattern[2])/count1[i-1]
    index_the = words.index('THE')
    prob_the = list(probability2[index_the, :].copy())
    print('the five most likely words follow THE')
    for i in range(5):
        ind = prob_the.index(max(prob_the))
        print(words[ind], max(prob_the))
        prob_the[ind] = 0
    # question3
    sentence = "Last week the stock market fell by one hundred points"
    [p1, p2] = sentence_probability(sentence, probability1, probability2, words)
    print('the log_probability of unigram model is', p1)
    print('the log_probability of bigram model is', p2)
    # question4
    sentence = "The nineteen officials sold fire insurance"
    [p1, p2] = sentence_probability(sentence, probability1, probability2, words)
    print('the log_probability of unigram model is', p1)
    print('the log_probability of bigram model is', p2)
    # question5
    x = np.linspace(0, 1, 200)
    y = []
    for i in x:
        y.append(sentence_probability_mix(sentence, probability1, probability2, words, i))
    plt.plot(x, y)
    plt.xlabel('λ')
    plt.ylabel('probability')
    plt.show()
    print('the optimal value of λ is %.2f' % x[y.index(max(y))])


if __name__ == "__main__":
    main()