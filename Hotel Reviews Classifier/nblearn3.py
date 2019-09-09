import json
import sys
import glob
import os
from collections import defaultdict
import string
import re
import math


stop_words = ['part', 'during', 'amount', 'becomes', 'and', 'etc', 'an', 'across', 'see', 'those', 'yourselves', 'our', 'seemed', 'without', 'through', 'empty', 'latterly', 'himself', 'herself', 'someone', 'has', 'interest', 'thereafter', 'whose', 'themselves', 'thru', 'why', 'being', 'still', 'whenever', 'be', 'de', 'call', 'detail', 'mine', 'beyond', 'find', 'once', 'fire', 'others', 'show', 'per', 'twenty', 'itself', 'do', 'either', 'only', 'eight', 'what', 'whatever', 'for', 'off', 'many', 'fifteen', 'hereupon', 'ltd', 're', 'eleven', 'indeed', 'about', 'more', 'my', 'nevertheless', 'behind', 'whether', 'sixty', 'whoever', 'whom', 'back', 'his', 'nothing', 'were', 'system', 'thus', 'here', 'much', 'couldnt', 'amoungst', 'due', 'now', 'get', 'nowhere', 'please', 'well', 'five', 'too', 'this', 'first', 'when', 'yourself', 'put', 'seeming', 'latter', 'as', 'may', 'hereby', 'me', 'ever', 'if', 'who', 'third', 'least', 'hundred', 'each', 'anyway', 'co', 'it', 'un', 'however', 'of', 'will', 'whence', 'i', 'often', 'side', 'whereafter', 'else', 'afterwards', 'thence', 'anywhere', 'whole', 'beside', 'one', 'towards', 'inc', 'can', 'whereas', 'down', 'fill', 'own', 'is', 'ourselves', 'so', 'six', 'up', 'thin', 'except', 'thereby', 'always', 'another', 'elsewhere', 'became', 'give', 'same', "can't", "needn't", "couldn't", "wouldn't", 'than', 'seem', 'cry', 'onto', 'throughout', 'around', 'anything', 'along', 'hers', 'almost', 'none', 'she', 'since', 'you', 'serious', 'was', 'yet', 'there', 'wherever', 'again', 'he', 'had', 'somewhere', 'everyone', 'cant', 'full', 'herein', 'though', 'her', 'then', 'nor', 'should', 'myself', 'hasnt', "hasn't", 'nobody', 'upon', 'done', 'formerly', 'hence', 'sometime', 'your', 'above', 'thereupon', 'beforehand', 'under', 'at', 'name', 'to', 'found', 'bottom', 'namely', 'further', 'all', 'become', 'few', 'describe', 'ie', 'even', 'whereupon', 'with', 'within', 'con', 'where', 'or', 'thick', 'some', 'are', 'via', 'go', 'wherein', 'together', 'mill', 'against', 'several', 'everything', 'while', 'ours', 'noone', 'something', 'also', 'other', 'seems', 'been', 'how', 'below', 'twelve', 'enough', 'the', 'four', 'next', 'a', 'made', 'every', 'in', 'otherwise', 'after', 'becoming', 'must', 'they', 'we', 'two', 'yours', 'no', 'whither', 'until', 'between', 'most', 'because', 'ten', 'meanwhile', 'amongst', 'but', 'into', 'less', 'perhaps', 'former', 'among', 'besides', 'forty', 'could', 'rather', 'although', 'from', 'such', 'very', 'take', 'sincere', 'him', 'neither', 'by', 'moreover', 'not', 'on', 'might', 'their', 'these', 'keep', 'therein', 'toward', 'both', 'over', 'whereby', 'would', 'never', 'everywhere', 'cannot', 'move', 'fifty', 'anyhow', 'its', 'sometimes', 'us', 'eg', 'out', 'top', 'alone', 'that', 'therefore', 'somehow', 'mostly', 'three', 'them', 'which', 'already', 'anyone', 'bill', 'last', 'hereafter', 'am', 'before', 'have', 'front', 'nine', 'any']


# Remove punctuation marks from each review
def remove_punctuation(review):
    return review.translate(str.maketrans('', '', string.punctuation))


def remove_stop_words(each_line):
    line_without_stop_words = []
    for w in each_line.split():
        if w not in stop_words:
            line_without_stop_words.append(w)
    return ' '.join(line_without_stop_words)


# Add a new word token to dictionary or update count of existing token
def update_wc(key, w):
    if w not in dictionaries[key]:
        dictionaries[key][w] = 1
    else:
        dictionaries[key][w] = dictionaries[key][w] + 1


# Apply add one LaPlace smoothing
def calc_conditional_probability(key, w, divisor):
    if w not in dictionaries[key]:
        dictionaries[key][w] = 1 / divisor
    else:
        dictionaries[key][w] = (dictionaries[key][w] + 1) / divisor
    dictionaries[key][w] = math.log(dictionaries[key][w])


# List all files, given the root of training data.


all_files = glob.glob(os.path.join(sys.argv[1], '*/*/*/*.txt'))

test_by_class = defaultdict(list)
train_by_class = defaultdict(list)


for f in all_files:
    # Take only last 4 components of the path. The earlier components are useless
    # as they contain path to the count_of_each_class directories.
    # class1, class2, fold, fname = f.split('/')[-4:]
    class1, class2, fold, fname = f.split('\\')[-4:]

    if fold == 'fold1':
        # True-clause will not enter in Vocareum as fold1 wont exist, but useful for your own code.
        test_by_class[class1+class2].append(f)
    else:
        train_by_class[class1+class2].append(f)

#
# TRAINMULTINOMIALNB(C,D)
# 1 V ← EXTRACTVOCABULARY(D)
# 2 N ← COUNTDOCS(D)
# 3 for each c ∈ C
# 4 do Nc ← COUNTDOCSINCLASS(D, c)
# 5 prior[c] ← Nc/N
# 6 textc ← CONCATENATETEXTOFALLDOCSINCLASS(D, c)
# 7 for each t ∈ V
# 8 do Tct ← COUNTTOKENSOFTERM(textc, t)
# 9 for each t ∈ V
# 10 do condprob[t][c] ← Tct+1
# åt′ (Tct′+1)
# 11 return V, prior, condprob
#
# APPLYMULTINOMIALNB(C,V, prior, condprob, d)
# 1 W ← EXTRACTTOKENSFROMDOC(V, d)
# 2 for each c ∈ C
# 3 do score[c] ← log prior[c]
# 4 for each t ∈ W
# 5 do score[c] += log condprob[t][c]
# 6 return argmaxc∈C score[c]
#

dictionaries = {'positive_polaritytruthful_from_TripAdvisor': {},
                'negative_polaritytruthful_from_Web': {},
                'positive_polaritydeceptive_from_MTurk': {},
                'negative_polaritydeceptive_from_MTurk': {}
                }

# count of each class
count_of_each_class = dict.fromkeys(dictionaries.keys(), 0)

# prior probabilities
prior_probabilities = dict.fromkeys(count_of_each_class.keys(), 0.0)

unique_words = set()

for k, v in train_by_class.items():

    for filename in v:
        f = open(filename, 'r')
        for line in f:
            # Remove Whitespaces, Punctuations, Numbers from line
            line = ' '.join(line.split())
            line = remove_punctuation(re.sub(r'\d+', '', line.lower()))
            line = remove_stop_words(line)

            # update prior probabilities of each class = Nc
            count_of_each_class[k] += 1

            for word in line.split():
                update_wc(k, word)
                unique_words.add(word)

# Find number of words in each class
denominator = [sum(dictionaries['positive_polaritytruthful_from_TripAdvisor'].values()), sum(dictionaries['negative_polaritytruthful_from_Web'].values()), sum(dictionaries['positive_polaritydeceptive_from_MTurk'].values()), sum(dictionaries['negative_polaritydeceptive_from_MTurk'].values())]

for word in unique_words:
    calc_conditional_probability('positive_polaritytruthful_from_TripAdvisor', word, denominator[0] + len(unique_words))
    calc_conditional_probability('negative_polaritytruthful_from_Web', word, denominator[1] + len(unique_words))
    calc_conditional_probability('positive_polaritydeceptive_from_MTurk', word, denominator[2] + len(unique_words))
    calc_conditional_probability('negative_polaritydeceptive_from_MTurk', word, denominator[3] + len(unique_words))

# Update Nc
count_of_each_class['positive_polaritytruthful_from_TripAdvisor'] = denominator[0]
count_of_each_class['negative_polaritytruthful_from_Web'] = denominator[1]
count_of_each_class['positive_polaritydeceptive_from_MTurk'] = denominator[2]
count_of_each_class['negative_polaritydeceptive_from_MTurk'] = denominator[3]

# Calculate log P(C) for each class - log Nc / N

num_documents = count_of_each_class['positive_polaritytruthful_from_TripAdvisor'] + count_of_each_class['negative_polaritytruthful_from_Web'] + count_of_each_class['positive_polaritydeceptive_from_MTurk'] + count_of_each_class['negative_polaritydeceptive_from_MTurk']
print(num_documents)
prior_probabilities['positive_polaritytruthful_from_TripAdvisor'] = math.log(count_of_each_class['positive_polaritytruthful_from_TripAdvisor'] / num_documents)
prior_probabilities['negative_polaritytruthful_from_Web'] = math.log(count_of_each_class['negative_polaritytruthful_from_Web'] / num_documents)
prior_probabilities['positive_polaritydeceptive_from_MTurk'] = math.log(count_of_each_class['positive_polaritydeceptive_from_MTurk'] / num_documents)
prior_probabilities['negative_polaritydeceptive_from_MTurk'] = math.log(count_of_each_class['negative_polaritydeceptive_from_MTurk'] / num_documents)


# Write the model file
with open('nbmodel.txt', 'w') as file:
    model_list = [prior_probabilities, dictionaries, len(unique_words), count_of_each_class]
    json.dump(model_list, file, indent=2)
