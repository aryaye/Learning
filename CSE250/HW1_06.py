# -*- coding: utf-8 -*-
import re
import operator


class Word6:
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
        p = p + word.if_match_evidence(words_in, words_out)*word.count/sum
    return p


def calculate_pwe(word, pe, sum, words_in, words_out):
    pwe = word.if_match_evidence(words_in, words_out)*(word.count/sum)/pe
    return pwe


def calculate_ple(word_list, l):
    ple = 0
    for word in word_list:
        ple = ple + word.if_exist_l(l)*word.pwe
    return ple


def main():
    # read the 5_letter words from txt and manage them
    f = open(r"C:\\Users\Haiya Ye\PycharmProjects\learn algorithms\CSE250\hw1_word_counts_06.txt","r")
    word_list = []
    line = f.readline()
    while line:
        count = int(re.findall("[0-9]+", line)[0])
        name = re.findall("[A-Z]+", line)[0]
        word_list.append(Word6(name, count))
        line = f.readline()
    f.close()

    word_list_sorted = sorted(word_list, key=operator.attrgetter("count"))
    # 14th least frequent 5-letter words
    print(word_list_sorted[13])
    # 15th most frequent 5-letter words
    print(word_list_sorted[len(word_list_sorted)-16])

    # evidence
    words_in = ["", "", "I", "", "E", ""]
    words_out = ["D"]

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
            ple_list.append(0)
        else:
            ple_list.append(calculate_ple(word_list, letter))

    print(chr(ple_list.index(max(ple_list))+65), max(ple_list))

    for i in range(26):
        print(chr(i+65), ple_list[i])


if __name__ == "__main__":
    main()
