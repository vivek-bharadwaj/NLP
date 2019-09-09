import sys
import os
import glob
import json
from collections import defaultdict
import string
import re
import math


stop_words = ['part', 'during', 'amount', 'becomes', 'and', 'etc', 'an', 'across', 'see', 'those', 'yourselves', 'our', 'seemed', 'without', 'through', 'empty', 'latterly', 'himself', 'herself', 'someone', 'has', 'interest', 'thereafter', 'whose', 'themselves', 'thru', 'why', 'being', 'still', 'whenever', 'be', 'de', 'call', 'detail', 'mine', 'beyond', 'find', 'once', 'fire', 'others', 'show', 'per', 'twenty', 'itself', 'do', 'either', 'only', 'eight', 'what', 'whatever', 'for', 'off', 'many', 'fifteen', 'hereupon', 'ltd', 're', 'eleven', 'indeed', 'about', 'more', 'my', 'nevertheless', 'behind', 'whether', 'sixty', 'whoever', 'whom', 'back', 'his', 'nothing', 'were', 'system', 'thus', 'here', 'much', 'couldnt', 'amoungst', 'due', 'now', 'get', 'nowhere', 'please', 'well', 'five', 'too', 'this', 'first', 'when', 'yourself', 'put', 'seeming', 'latter', 'as', 'may', 'hereby', 'me', 'ever', 'if', 'who', 'third', 'least', 'hundred', 'each', 'anyway', 'co', 'it', 'un', 'however', 'of', 'will', 'whence', 'i', 'often', 'side', 'whereafter', 'else', 'afterwards', 'thence', 'anywhere', 'whole', 'beside', 'one', 'towards', 'inc', 'can', 'whereas', 'down', 'fill', 'own', 'is', 'ourselves', 'so', 'six', 'up', 'thin', 'except', 'thereby', 'always', 'another', 'elsewhere', 'became', 'give', 'same', "can't", "needn't", "couldn't", "wouldn't", 'than', 'seem', 'cry', 'onto', 'throughout', 'around', 'anything', 'along', 'hers', 'almost', 'none', 'she', 'since', 'you', 'serious', 'was', 'yet', 'there', 'wherever', 'again', 'he', 'had', 'somewhere', 'everyone', 'cant', 'full', 'herein', 'though', 'her', 'then', 'nor', 'should', 'myself', 'hasnt', "hasn't", 'nobody', 'upon', 'done', 'formerly', 'hence', 'sometime', 'your', 'above', 'thereupon', 'beforehand', 'under', 'at', 'name', 'to', 'found', 'bottom', 'namely', 'further', 'all', 'become', 'few', 'describe', 'ie', 'even', 'whereupon', 'with', 'within', 'con', 'where', 'or', 'thick', 'some', 'are', 'via', 'go', 'wherein', 'together', 'mill', 'against', 'several', 'everything', 'while', 'ours', 'noone', 'something', 'also', 'other', 'seems', 'been', 'how', 'below', 'twelve', 'enough', 'the', 'four', 'next', 'a', 'made', 'every', 'in', 'otherwise', 'after', 'becoming', 'must', 'they', 'we', 'two', 'yours', 'no', 'whither', 'until', 'between', 'most', 'because', 'ten', 'meanwhile', 'amongst', 'but', 'into', 'less', 'perhaps', 'former', 'among', 'besides', 'forty', 'could', 'rather', 'although', 'from', 'such', 'very', 'take', 'sincere', 'him', 'neither', 'by', 'moreover', 'not', 'on', 'might', 'their', 'these', 'keep', 'therein', 'toward', 'both', 'over', 'whereby', 'would', 'never', 'everywhere', 'cannot', 'move', 'fifty', 'anyhow', 'its', 'sometimes', 'us', 'eg', 'out', 'top', 'alone', 'that', 'therefore', 'somehow', 'mostly', 'three', 'them', 'which', 'already', 'anyone', 'bill', 'last', 'hereafter', 'am', 'before', 'have', 'front', 'nine', 'any']


def remove_punctuation(review):
    return review.translate(str.maketrans('', '', string.punctuation))


def remove_stop_words(each_line):
    line_without_stop_words = []
    for w in each_line.split():
        if w not in stop_words:
            line_without_stop_words.append(w)
    return ' '.join(line_without_stop_words)


def calc_conditional_probability(key, w):
    if w in conditional_probabilities[key]:
        return conditional_probabilities[key][w]
    else:
        return math.log(1 / unique_words_count + classes[key])


# APPLYMULTINOMIALNB(C,V, prior, condprob, d)
# W ← EXTRACTTOKENSFROMDOC(V, d)
# for each c ∈ C
# do score[c] ← log prior[c]
# for each t ∈ W
# do score[c] += log condprob[t][c]
# return argmaxc∈C score[c]
#


def classify_review(each_line):
    score['positive_polaritytruthful_from_TripAdvisor'] = prior_probabilities['positive_polaritytruthful_from_TripAdvisor']
    score['negative_polaritytruthful_from_Web'] = prior_probabilities['negative_polaritytruthful_from_Web']
    score['positive_polaritydeceptive_from_MTurk'] = prior_probabilities['positive_polaritydeceptive_from_MTurk']
    score['negative_polaritydeceptive_from_MTurk'] = prior_probabilities['negative_polaritydeceptive_from_MTurk']

    each_line = remove_punctuation(re.sub(r'\d+', '', each_line.lower()))
    each_line = remove_stop_words(each_line)
    each_line = ' '.join(each_line.split())

    for word in each_line.split():
        score['positive_polaritytruthful_from_TripAdvisor'] += calc_conditional_probability('positive_polaritytruthful_from_TripAdvisor', word)
        score['negative_polaritytruthful_from_Web'] += calc_conditional_probability('negative_polaritytruthful_from_Web', word)
        score['positive_polaritydeceptive_from_MTurk'] += calc_conditional_probability('positive_polaritydeceptive_from_MTurk', word)
        score['negative_polaritydeceptive_from_MTurk'] += calc_conditional_probability('negative_polaritydeceptive_from_MTurk', word)

    max_key = max(score, key=score.get)
    return max_key


all_files = glob.glob(os.path.join(sys.argv[1], '*/*/*/*.txt'))
test_by_class = defaultdict(list)

for f in all_files:
    class1, class2, fold, fname = f.split('/')[-4:]
    test_by_class[class1+class2].append(f)

model_file = 'nbmodel.txt'
parsed_file = json.load(open(model_file))

prior_probabilities = parsed_file[0]
conditional_probabilities = parsed_file[1]
unique_words_count = parsed_file[2]
classes = parsed_file[3]

score = dict.fromkeys(classes)

with open('nboutput.txt', 'w') as output_file:
    for k, v in test_by_class.items():
        for file in v:
            input_file = open(file, 'r')
            for line in input_file:
                class_name = classify_review(line.strip())
                if class_name == 'positive_polaritytruthful_from_TripAdvisor':
                    output_file.write('truthful' + ' ' + 'positive' + ' ' + input_file.name + '\n')
                elif class_name == 'negative_polaritytruthful_from_Web':
                    output_file.write('truthful' + ' ' + 'negative' + ' ' + input_file.name + '\n')
                elif class_name == 'positive_polaritydeceptive_from_MTurk':
                    output_file.write('deceptive' + ' ' + 'positive' + ' ' + input_file.name + '\n')
                elif class_name == 'negative_polaritydeceptive_from_MTurk':
                    output_file.write('deceptive' + ' ' + 'negative' + ' ' + input_file.name + '\n')
