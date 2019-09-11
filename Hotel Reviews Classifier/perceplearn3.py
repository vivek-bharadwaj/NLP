# python perceplearn3.py /path/to/input
# write the model parameters to two files: vanillamodel.txt for the vanilla perceptron,
# and averagedmodel.txt for the averaged perceptron.

import os
import glob
import sys
import json
import re
import string
from collections import Counter, defaultdict
from random import shuffle
import numpy as np

stop_words = ['part', 'during', 'amount', 'becomes', 'and', 'etc', 'an', 'across', 'see', 'those', 'yourselves', 'our',
              'seemed', 'without', 'through', 'empty', 'latterly', 'himself', 'herself', 'someone', 'has', 'interest',
              'thereafter', 'whose', 'themselves', 'thru', 'why', 'being', 'still', 'whenever', 'be', 'de', 'call',
              'detail', 'mine', 'beyond', 'find', 'once', 'fire', 'others', 'show', 'per', 'twenty', 'itself', 'do',
              'either', 'only', 'eight', 'what', 'whatever', 'for', 'off', 'many', 'fifteen', 'hereupon', 'ltd', 're',
              'eleven', 'indeed', 'about', 'more', 'my', 'nevertheless', 'behind', 'whether', 'sixty', 'whoever',
              'whom', 'back', 'his', 'nothing', 'were', 'system', 'thus', 'here', 'much', 'couldnt', 'amoungst', 'due',
              'now', 'get', 'nowhere', 'please', 'well', 'five', 'too', 'this', 'first', 'when', 'yourself', 'put',
              'seeming', 'latter', 'as', 'may', 'hereby', 'me', 'ever', 'if', 'who', 'third', 'least', 'hundred',
              'each', 'anyway', 'co', 'it', 'un', 'however', 'of', 'will', 'whence', 'i', 'often', 'side', 'whereafter',
              'else', 'afterwards', 'thence', 'anywhere', 'whole', 'beside', 'one', 'towards', 'inc', 'can', 'whereas',
              'down', 'fill', 'own', 'is', 'ourselves', 'so', 'six', 'up', 'thin', 'except', 'thereby', 'always',
              'another', 'elsewhere', 'became', 'give', 'same', "can't", "needn't", "couldn't", "wouldn't", 'than',
              'seem', 'cry', 'onto', 'throughout', 'around', 'anything', 'along', 'hers', 'almost', 'none', 'she',
              'since', 'you', 'serious', 'was', 'yet', 'there', 'wherever', 'again', 'he', 'had', 'somewhere',
              'everyone', 'cant', 'full', 'herein', 'though', 'her', 'then', 'nor', 'should', 'myself', 'hasnt',
              "hasn't", 'nobody', 'upon', 'done', 'formerly', 'hence', 'sometime', 'your', 'above', 'thereupon',
              'beforehand', 'under', 'at', 'name', 'to', 'found', 'bottom', 'namely', 'further', 'all', 'become', 'few',
              'describe', 'ie', 'even', 'whereupon', 'with', 'within', 'con', 'where', 'or', 'thick', 'some', 'are',
              'via', 'go', 'wherein', 'together', 'mill', 'against', 'several', 'everything', 'while', 'ours', 'noone',
              'something', 'also', 'other', 'seems', 'been', 'how', 'below', 'twelve', 'enough', 'the', 'four', 'next',
              'a', 'made', 'every', 'in', 'otherwise', 'after', 'becoming', 'must', 'they', 'we', 'two', 'yours', 'no',
              'whither', 'until', 'between', 'most', 'because', 'ten', 'meanwhile', 'amongst', 'but', 'into', 'less',
              'perhaps', 'former', 'among', 'besides', 'forty', 'could', 'rather', 'although', 'from', 'such', 'very',
              'take', 'sincere', 'him', 'neither', 'by', 'moreover', 'not', 'on', 'might', 'their', 'these', 'keep',
              'therein', 'toward', 'both', 'over', 'whereby', 'would', 'never', 'everywhere', 'cannot', 'move', 'fifty',
              'anyhow', 'its', 'sometimes', 'us', 'eg', 'out', 'top', 'alone', 'that', 'therefore', 'somehow', 'mostly',
              'three', 'them', 'which', 'already', 'anyone', 'bill', 'last', 'hereafter', 'am', 'before', 'have',
              'front', 'nine', 'any']


def remove_punctuation(review):
    return review.translate(str.maketrans('', '', string.punctuation))


def remove_stop_words(each_line):
    line_without_stop_words = []
    for w in each_line.split():
        if w not in stop_words:
            line_without_stop_words.append(w)
    return ' '.join(line_without_stop_words)


train_by_class = defaultdict(list)  # {"class1+class2":[list of files]}
feature_vector_td = {}  # {filename : {word:count}} for truthful deceptive
feature_vector_pn = {}  # {filename : {word:count}} for positive negative
y_true_pn = {}  # y_true_pn = {filename : +1/-1}
y_true_td = {}  # y_true_td = {filename : +1/-1}
weight_vector_td = defaultdict(float)  # weights for truthful deceptive classification
weight_vector_pn = defaultdict(float)  # weights for positive negative classification
cached_weight_vector_td = defaultdict(float)  # for averaged perceptron
cached_weight_vector_pn = defaultdict(float)  # for averaged perceptron
bias = [0, 0]  # [0] is for td, [1] is for pn
cached_bias = [0, 0]  # for averaged perceptron [0] is for td, [1] is for pn
unique_words = set()  # set of words in vocabulary
counts = {}  # total words count {word:count}


def get_true_class_of_td(fname):
    if fname.find("truthful") != -1:
        return 1
    return -1


def get_true_class_of_pn(fname):
    if fname.find("positive") != -1:
        return 1
    return -1


def train_perceptron(w, x, y_true, b, u, beta, averaged=True):  # y_true[filename] = true_
    max_iter = 70
    np.random.seed(100)
    filenames_list = list(x.keys())
    shuffle(filenames_list)
    c = 1
    for i in range(1, max_iter):
        # permute()
        shuffle(filenames_list)
        for filename in filenames_list:
            y_pred = 0
            word_freq = x[filename]
            for word, count_of_word in word_freq.items():
                if word not in unique_words:
                    continue
                y_pred += w[word] * count_of_word

            if y_true[filename] * (y_pred + b) <= 0:
                for word, count_of_word in word_freq.items():
                    if word not in unique_words:
                        continue
                    w[word] += y_true[filename] * count_of_word
                    # print(str(word) + " : " + str(count_of_word) + " : " + str(w[word]))
                    if averaged:
                        u[word] += y_true[filename] * count_of_word * c
                b += y_true[filename]
                if averaged:
                    beta += y_true[filename] * c
            c += 1
        if averaged:
            for key, value in w.items():
                u[key] = value - (1 / c) * u[key]
            beta = b - (1 / c) * beta
    return w, b, u, beta


# expected output dictionary
# output = {'positive_polaritydeceptive_from_MTurk': 1.0,
#           'positive_polaritytruthful_from_TripAdvisor': 1.0,
#           'negative_polaritydeceptive_from_MTurk': -1.0,
#           'negative_polaritytruthful_from_Web': -1.0
#           }

all_files = glob.glob(os.path.join(sys.argv[1], '*/*/*/*.txt'))

for f in all_files:
    class1, class2, fold, filename = f.split('\\')[-4:] # Local env
    # class1, class2, fold, filename = f.split('/')[-4:] For Vocareum
    y_true_pn[filename] = get_true_class_of_pn(class1)
    y_true_td[filename] = get_true_class_of_td(class2)
    train_by_class[class1 + class2].append(f)

all_reviews = []
for k, v in train_by_class.items():
    for filename in v:
        f = open(filename, 'r')
        for line in f:
            line = ' '.join(line.split())
            line = remove_punctuation(re.sub(r'\d+', '', line.lower()))
            line = remove_stop_words(line)
            # print(line)
            all_reviews += line.split()
            feature_vector_pn[filename] = Counter(line.split())
            feature_vector_td[filename] = Counter(line.split())

            for word in line.split():
                unique_words.add(word)

            pn_val = -1
            td_val = -1

            if filename.find("positive") != -1:
                pn_val = 1
            if filename.find("truthful") != -1:
                td_val = 1

            y_true_pn[filename] = pn_val
            y_true_td[filename] = td_val

counts = Counter(all_reviews)
total = sum(counts.values())

for word in counts:
    if counts[word] < 5:
        unique_words.remove(word)

# print(len(unique_words))

# print(json.dumps(y_true_pn,indent=2))
weight_vector_td, bias[0], cached_weight_vector_td, cached_bias[0] = train_perceptron(weight_vector_td,
                                                                                      feature_vector_td, y_true_td,
                                                                                      bias[0], cached_weight_vector_td,
                                                                                      cached_bias[0], averaged=True)
weight_vector_pn, bias[1], cached_weight_vector_pn, cached_bias[1] = train_perceptron(weight_vector_pn,
                                                                                      feature_vector_pn, y_true_pn,
                                                                                      bias[1], cached_weight_vector_pn,
                                                                                      cached_bias[1], averaged=True)

with open('vanillamodel.txt', 'w') as file:
    model_list = [weight_vector_td, weight_vector_pn, bias, counts]
    json.dump(model_list, file, indent=2)

with open('averagedmodel.txt', 'w') as file:
    model_list = [cached_weight_vector_td, cached_weight_vector_pn,
                  cached_bias, counts]
    json.dump(model_list, file, indent=2)

# print(len(weight_vector_td))
# print(len(weight_vector_pn))
# print(cached_weight_vector_td)
# print(cached_weight_vector_pn)
#
# print(bias)
# print(cached_bias)
