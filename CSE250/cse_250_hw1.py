import urllib.request
import random
from sklearn import svm


def parseData(fname):
  for l in urllib.request.urlopen(fname):
    yield eval(l)

print("Reading data...")
data = list(parseData("http://jmcauley.ucsd.edu/cse255/data/beer/beer_50000.json"))
print("done")

# split random training set

print("spliting ... ")
random.shuffle(data)
training_data_r = data[:25000]
test_data_r = data[25000:]
print("done")

def balanced_class(datas):
    resp = []
    count = 0
    for d in datas:
        if count < 25000:
            if d['beer/style'] == "Hefeweizen":
                for i in range(40):
                    resp.append(d)
                    count += 1
            else:
                resp.append(d)
                count += 1
    return random.sample(resp, 2500)


training_data_r_balanced = balanced_class(training_data_r)

# classification
# 1.7. train SVM classifier in random split training data
print("# 1.7. train SVM classifier in random split training data")


def feature3(datum):
    if datum['beer/style'] == "Hefeweizen":
        return 1
    else:
        return 0


def feature4(d):
    return [d['review/taste'], d['review/appearance'], d['review/aroma'], d['review/palate'], d['review/overall']]


train_X = [feature4(d) for d in training_data_r_balanced]
train_y = [feature3(d) for d in training_data_r_balanced]

test_X = [feature4(d) for d in test_data_r]
test_y = [feature3(d) for d in test_data_r]

print('training...')
clf = svm.SVC(C=1000, kernel='rbf')
clf.fit(train_X, train_y)
print('done')

print('predicting ...')
train_predictions = clf.predict(train_X)
print('done')
train_accurarcy = sum(train_predictions == train_y)*1.0/len(train_X)
print('training_accuracy: %.5f'% train_accurarcy)

print('predicting...')
test_predictions = clf.predict(test_X)
print('done')
test_accurarcy = sum(test_predictions == test_y)*1.0/len(test_X)
print('test_accuracy: %.5f'% test_accurarcy)

# 1.8 more accurate predictor
print("# 1.8. more accurate predictor considering the text")


def feature5(d):
    resp = []
    if "banana" in d["review/text"]:
        resp.append(1)
    else:
        resp.append(0)
    resp.append(d['review/taste'])
    resp.append(d['review/appearance'])
    resp.append(d['review/aroma'])
    resp.append(d['review/palate'])
    #   resp.append(d['review/overall'])
    return resp


train_X = [feature5(d) for d in training_data_r_balanced]
train_y = [feature3(d) for d in training_data_r_balanced]

test_X = [feature5(d) for d in test_data_r]
test_y = [feature3(d) for d in test_data_r]

print('training...')
clf = svm.SVC(C=1000, kernel='rbf')
clf.fit(train_X, train_y)
print("done")

print('predicting ...')
train_predictions = clf.predict(train_X)
print('done')
train_accurarcy = sum(train_predictions == train_y)*1.0/len(train_X)
print('training_accuracy: %.5f'% train_accurarcy)

print('predicting...')
test_predictions = clf.predict(test_X)
print('done')
test_accurarcy = sum(test_predictions == test_y)*1.0/len(test_X)
print('test_accuracy: %.5f'% test_accurarcy)

