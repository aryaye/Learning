import urllib.request
from collections import defaultdict, Counter
import string
from sklearn import linear_model
from math import log10


class Reviews:
    def __init__(self):
        self.words = defaultdict(int)
        self.biwords = defaultdict(int)
        self.reviews = {}
        self.nreview = 0
        self.idf = defaultdict(float)
        self.biwords_idx = {}
        self.words_idx = {}
        self.word_length = [0]*2
        self.uniwords1000 = {}
        self.biwords1000 = {}

    def add(self, sentence):
        punctuation = set(string.punctuation)
        sentence = ''.join([c for c in sentence.lower() if c not in punctuation]).split()
        if not sentence:
            self.reviews[self.nreview] = ['', '']
            self.nreview += 1
            return
        self.reviews[self.nreview] = [sentence]
        idx = -1
        temp = []
        for idx in range(len(sentence)-1):
            temp.append(sentence[idx]+' '+sentence[idx+1])
            self.biwords[sentence[idx]+' '+sentence[idx+1]] += 1
            self.words[sentence[idx]] += 1
        self.words[sentence[idx+1]] += 1
        self.reviews[self.nreview].append(temp)
        self.nreview += 1

    def add_data(self, data, name):
        for d in data:
            self.add(d[name])

    def form_word_idx(self, t):
        # t = 1: unigram; t = 2: bigram
        if t == 1:
            self.word_length[0] = len(self.words)
            self.words_idx = dict(zip(self.words.keys(), range(self.word_length[0])))
        elif t == 2:
            self.word_length[1] = len(self.biwords)
            self.biwords_idx = dict(zip(self.biwords.keys(), range(self.word_length[1])))

    def output_most_frequency(self, k, t):
        # output the most frequency words
        if t == 1:
            temp = [[x, y] for x, y in zip(self.words.values(), self.words.keys())]
        elif t == 2:
            temp = [[x, y] for x, y in zip(self.biwords.values(), self.biwords.keys())]
        else:
            return []
        temp.sort()
        ans = []
        for x, y in temp[::-1][:k]:
            ans.append(y)
        return ans

    def bi_count_regressions(self, k, data, lamda):
        words = self.output_most_frequency(k, 2)
        wordId = dict(zip(words, range(len(words))))

        def feature(datum):
            feat = [0] * len(words)
            r = ''.join([c for c in datum['review/text'].lower() if c not in string.punctuation]).split()
            for idx in range(len(r) - 1):
                w = r[idx]+' '+r[idx + 1]
                if w in words:
                    feat[wordId[w]] += 1
            feat.append(1)  # offset
            return feat

        x = [feature(d) for d in data]
        y = [d['review/overall'] for d in data]
        clf = linear_model.Ridge(lamda, fit_intercept=False)
        clf.fit(x, y)
        theta = clf.coef_
        predictions = clf.predict(x)
        return wordId, theta, predictions

    def words_vector(self):
        words = self.output_most_frequency(1000, 1)
        self.uniwords1000 = dict(zip(words, range(len(words))))
        words = self.output_most_frequency(1000, 2)
        self.biwords1000 = dict(zip(words, range(len(words))))

    def uni_tfidf_regression(self, k, data, lamda):
        words = self.output_most_frequency(k, 1)
        wordId = dict(zip(words, range(len(words))))

        def feature(datum):
            feat = [0] * len(words)
            r = ''.join([c for c in datum['review/text'].lower() if c not in string.punctuation]).split()
            dicts = Counter(r)
            for w in set(r):
                if w in words:
                    feat[wordId[w]] = self.idf[w] * dicts[w]
            feat.append(1)  # offset
            return feat

        x = [feature(d) for d in data]
        y = [d['review/overall'] for d in data]
        clf = linear_model.Ridge(lamda, fit_intercept=False)
        clf.fit(x, y)
        theta = clf.coef_
        predictions = clf.predict(x)
        return wordId, theta, predictions

    def calculate_idf(self):
        # calculate tf-idf of kth sentence, t words as a term, t is 1 or 2
        for s in self.uniwords1000:
            for review in self.reviews.values():
                if s in review[0]:
                    self.idf[s] += 1.0
            self.idf[s] = log10(self.nreview/self.idf[s])
        for s in self.biwords1000:
            for review in self.reviews.values():
                if s in review[1]:
                    self.idf[s] += 1.0
            self.idf[s] = log10(self.nreview/self.idf[s])

    def calculate_tfidf(self, word_list, d):
        # d is a dict of tf
        ans = []
        for word in word_list:
            ans.append([word, d[word] * self.idf[word]])
        return ans

    def calculate_sentence_tfidf(self, k, t):
        # calculate tf-idf of kth sentence, t words as a term, t is 1 or 2
        reviews = self.reviews[k][t-1]
        tfidf = [0]*1000
        for word in reviews:
            if t == 1 and word in self.uniwords1000:
                tfidf[self.uniwords1000[word]] += 1
            elif t == 2 and word in self.biwords1000:
                tfidf[self.biwords1000[word]] += 1
        if t == 1:
            for word in self.uniwords1000:
                tfidf[self.uniwords1000[word]] *= self.idf[word]
        elif t == 2:
            for word in self.biwords1000:
                tfidf[self.biwords1000[word]] *= self.idf[word]
        return tfidf


def parseData(fname):
    for l in urllib.request.urlopen(fname):
        yield eval(l)


def calculate_mse(predictions, ratings):
    ans = 0
    n = len(ratings)
    for k in range(n):
        ans += (ratings[k]-predictions[k])**2
    return ans/n


def cos_similarity(vector1, vector2):
    inner = [vector1[idx] * vector2[idx] for idx in range(len(vector1))]
    sum1 = sum([x**2 for x in vector1])**0.5
    sum2 = sum([x**2 for x in vector2])**0.5
    return sum(inner)/(sum1*sum2)


def main():
    print("Reading data...")
    data = list(parseData("http://jmcauley.ucsd.edu/cse190/data/beer/beer_50000.json"))[:5000]
    print("done")

    r = Reviews()
    r.add_data(data, 'review/text')
    r.form_word_idx(1)
    r.form_word_idx(2)
    r.words_vector()
    r.calculate_idf()
    # task1
    task1 = r.output_most_frequency(5, 2)
    print([[item, r.biwords[item]] for item in task1])
    # task2
    wordId, theta, predictions = r.bi_count_regressions(1000, data, 1)
    print('mse of bigram model:', calculate_mse(predictions, [d['review/overall'] for d in data]))
    # task3
    words_list = ['foam', 'smell', 'banana', 'lactic', 'tart']
    task3 = r.calculate_tfidf(words_list, Counter(r.reviews[0][0]))
    print([[x[0], r.idf[x[0]], x[1]] for x in task3])
    # task4
    vector1 = r.calculate_sentence_tfidf(0, 1)
    vector2 = r.calculate_sentence_tfidf(1, 1)
    print('the cosine similarity between the first and the second review is', cos_similarity(vector1, vector2))
    # task5
    comp = []
    for k in range(1, 5000):
        vector = r.calculate_sentence_tfidf(k, 1)
        comp.append([cos_similarity(vector1, vector), k])
    comp.sort()
    print(data[comp[-1][1]])
    # task6
    wordId, theta, predictions = r.uni_tfidf_regression(1000, data, 1)
    print('mse of unigram tf-idf model:', calculate_mse(predictions, [d['review/overall'] for d in data]))
    # task7
