# -*- coding: utf-8 -*-
import re
from flask import Flask, request, jsonify, make_response
from functools import wraps
import random

app = Flask(__name__)


def allow_cross_domain(fun):
    @wraps(fun)
    def wrapper_fun(*args, **kwargs):
        rst = make_response(fun(*args, **kwargs))
        rst.headers['Access-Control-Allow-Origin'] = '*'
 #       rst.headers['Access-Control-Allow-Methods'] = 'PUT,GET,POST,DELETE'
 #       allow_headers = "Referer,Accept,Origin,User-Agent"
 #       rst.headers['Access-Control-Allow-Headers'] = allow_headers
        return rst
    return wrapper_fun


@app.route('/word', methods = ['GET'])
@allow_cross_domain
def random_word():
    types = request.args['type']
    paths = r"hw1_word_counts_0" + str(types) + ".txt"
    f = open(paths,"r")
    word_list = []
    line = f.readline()
    while line:
        name = re.findall("[A-Z]+", line)[0]
        word_list.append(name)
        line = f.readline()
    f.close()
    return jsonify(data = word_list[random.randint(0, len(word_list))])


@app.route('/guess', methods = ['GET','POST'])
@allow_cross_domain
def possibility_of_next_guess():
    word = request.args['word']
    letters = request.args['letters']
    words_in = []
    words_out = []
    paths = r"hw1_word_counts_0" + str(len(word)) + ".txt"
    for cha in word:
        if cha in letters:
            words_in.append(cha)
        else:
            words_in.append('')
    for cha in letters:
        if cha not in word:
            words_out.append(cha)

    req = probability_of_next_guess(words_in, words_out,paths)
    req.append(words_in)
    req.append(words_out)
    return jsonify(data = req)



class Word:
    def __init__(self, name, count):
        self.name = name
        self.count = count
        self.pwe = 0

    def if_match_evidence(self, words_in, words_out):
        order = 0
        for letter in words_in:
            if self.name[order] in words_out:
                return 0
            if letter and letter != self.name[order]:
                return 0
            if (not letter) and self.name[order] in words_in:
                return 0
            order = order + 1
        return 1

    def if_exist_l(self, letter):
        if letter in self.name:
            return 1
        return 0

    def add_pwe(self, pwe):
        self.pwe = pwe


def calculate_pe(word_list, words_in, words_out, sum):
    p = 0
    for word in word_list:
        p = p + word.if_match_evidence(words_in, words_out)*word.count*1.0/sum
    return p


def calculate_pwe(word, pe, sum, words_in, words_out):
    pwe = word.if_match_evidence(words_in, words_out)*(word.count*1.0/sum)*1.0/pe
    return pwe


def calculate_ple(word_list, l):
    ple = 0
    for word in word_list:
        ple = ple + word.if_exist_l(l)*word.pwe
    return ple


def probability_of_next_guess(words_in, words_out,paths):
    # read the 5_letter words from txt and manage them
    f = open(paths,"r")
    word_list = []
    line = f.readline()
    while line:
        count = int(re.findall("[0-9]+", line)[0])
        name = re.findall("[A-Z]+", line)[0]
        word_list.append(Word(name, count))
        line = f.readline()
    f.close()

    ple_list = []
    sum = 0
    for word in word_list:
        sum = sum + word.count

    # calculate the P(E)
    pe = calculate_pe(word_list, words_in, words_out, sum)

    # calculate the P(W=w|E) of each word in word_list
    for word in word_list:
        word.add_pwe(calculate_pwe(word, pe, sum, words_in, words_out))

    # calculate the P(l|E) for each word in
    for i in range(65, 91):
        letter = chr(i)
        if letter in words_in or letter in words_out:
            ple_list.append(0.0)
        else:
            ple_list.append(calculate_ple(word_list, letter))
    print(ple_list)
    return ple_list


if __name__ == "__main__":
    app.run()
