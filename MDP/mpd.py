# -*- coding: utf-8 -*-
from collections import defaultdict
import random
from numpy import *
from PIL import Image, ImageDraw, ImageFont


class Mdp(object):
    def __init__(self, prob, reward, r, s_size):
        """
        a = 0, 1, 2, 3: west, north, east, south
        :param prob:  list of list of list: [[s, s', prob(s'|s,0)],[s, s', prob(s'|s,1)], ...]
        :param reward: list of float R(s)
        """
        self.r = r
        self.a_size = len(prob)
        self.s_size = s_size
        self.a = [a for a in range(self.a_size)]
        self.s = [s for s in range(1, self.s_size+1)]
        self.prob = []
        for i in range(self.a_size):
            self.prob.append(defaultdict(dict))
        for a in range(self.a_size):
            for s, s_plus, p in prob[a]:
                self.prob[a][s][s_plus] = p
        self.reward = reward

    def expect_long_turn_return(self, Pi):
        """
        :param Pi: list
        :return: matrix of size self.s x 1
        """
        P = matrix(zeros((self.s_size, self.s_size)))
        for s in self.s:
            a = Pi[s-1]
            for s_plus in self.prob[a][s]:
                P[s-1, s_plus-1] = self.prob[a][s][s_plus]*self.r
        V = (matrix(eye(self.s_size))-P).I*matrix(self.reward).T
        return V

    def greedy_policy_iteration(self):
        """
        :return: former: list of int: optimal policy
                 latter: matrix of float: optimal value function, size: self.s_size x 1
        """
        Pi_old = [random.sample(self.a, 1)[0] for _ in range(self.s_size)]
        while True:
            V = self.expect_long_turn_return(Pi_old)
            Pi_greedy = []
            for s in self.s:
                temp = []
                for a in self.a:
                    sums = 0
                    for s_plus in self.prob[a][s]:
                        sums += self.prob[a][s][s_plus]*V[s_plus-1, 0]
                    temp.append(sums)
                Pi_greedy.append(temp.index((max(temp))))
            if Pi_old == Pi_greedy:
                break
            Pi_old = Pi_greedy.copy()
        return Pi_greedy, self.expect_long_turn_return(Pi_greedy)

    def value_iteration(self):
        """
        calculate optimal value function by value iteration algorithm
        :return:  list of float
        """
        values_old = [0]*self.s_size
        while True:
            values_new = []
            for s in self.s:
                temp = []
                for a in self.a:
                    sums = 0
                    for s_plus in self.prob[a][s]:
                        sums += self.prob[a][s][s_plus]*values_old[s_plus-1]
                    temp.append(sums)
                values_new.append(self.reward[s-1]+self.r*max(temp))
            if diff(values_new, values_old) <= 0.00000001:
                break
            values_old = values_new.copy()
        return values_new


def diff(x1, x2):
    res = 0
    for a, b in zip(x1, x2):
        res += (a-b)**2
    return res


def add_number_to_picture(numbers, number_dict, orignalpath, resultPath):
    img = Image.open(orignalpath)
    draw = ImageDraw.Draw(img)
    myfont = ImageFont.truetype('C:/windows/fonts/Arial.ttf', size=30)
    fillcolor = "#000000"
    for idx, n in enumerate(numbers):
        if n > 0:
            col = idx%9
            row = idx//9
            draw.text((92*row, 92*col+20), number_dict[n], font=myfont, fill = fillcolor)
    img.show()
    img.save(resultPath)


def main():
    with open(r'C:\\Users\Haiya Ye\PycharmProjects\learn algorithms\MDP\prob_a1.txt', 'r') as f:
        prob_a1 = f.read().split('\n')[:-1]
    prob_a1 = [[int(x.split()[0]), int(x.split()[1]), float(x.split()[2])] for x in prob_a1]
    with open(r'C:\\Users\Haiya Ye\PycharmProjects\learn algorithms\MDP\prob_a2.txt', 'r') as f:
        prob_a2 = f.read().split('\n')[:-1]
    prob_a2 = [[int(x.split()[0]), int(x.split()[1]), float(x.split()[2])] for x in prob_a2]
    with open(r'C:\\Users\Haiya Ye\PycharmProjects\learn algorithms\MDP\prob_a3.txt', 'r') as f:
        prob_a3 = f.read().split('\n')[:-1]
    prob_a3 = [[int(x.split()[0]), int(x.split()[1]), float(x.split()[2])] for x in prob_a3]
    with open(r'C:\\Users\Haiya Ye\PycharmProjects\learn algorithms\MDP\prob_a4.txt', 'r') as f:
        prob_a4 = f.read().split('\n')[:-1]
    prob_a4 = [[int(x.split()[0]), int(x.split()[1]), float(x.split()[2])] for x in prob_a4]
    with open(r'C:\\Users\Haiya Ye\PycharmProjects\learn algorithms\MDP\rewards.txt', 'r') as f:
        reward = f.read().split('\n')[:-1]
    reward = [float(x) for x in reward]
    prob = [prob_a1, prob_a2, prob_a3, prob_a4]
    mdp = Mdp(prob, reward, 0.99, 81)

    # greedy policy iteration
    optimalPolicy, optimalValueFunction1 = mdp.greedy_policy_iteration()

    # draw optimal value function picture
    optimalv = [optimalValueFunction1[i,0] for i in range(mdp.s_size)]
    optimalv_dict = dict(zip(optimalv, [str(round(x, 2)) for x in optimalv]))
    add_number_to_picture(optimalv, optimalv_dict, 'MDP/ans1.PNG', 'MDP/result_optimalv.png')

    # draw optimal policy picture
    optimalp_dict = {
        0:'←',
        1:'↑',
        2:'→',
        3:'↓'
    }
    add_number_to_picture(optimalPolicy, optimalp_dict, 'MDP/ans1.PNG', 'MDP/result_optimalp.png')

    # value iteration
    optimalValueFunction2 = mdp.value_iteration()
    optimalv2_dict = dict(zip(optimalValueFunction2, [str(round(x, 2)) for x in optimalValueFunction2]))
    add_number_to_picture(optimalValueFunction2, optimalv2_dict, 'MDP/ans1.PNG', 'MDP/result_optimalv2.png')


if __name__ == '__main__':
    main()


