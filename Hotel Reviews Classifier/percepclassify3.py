import sys
import glob
import os
from collections import Counter, defaultdict
import string
import json
import re

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


def classify(review, W, bias):
    # print(json.dumps(W,indent=2), bias)
    review_file = open(review, 'r')
    pred = 1
    for line in review_file:
        line = ' '.join(line.split())
        line = remove_punctuation(re.sub(r'\d+', '', line.lower()))
        line = remove_stop_words(line)
        word_freq = Counter(line.split())
        # calculate pred
        pred = bias
        for w, f in word_freq.items():
            if w in vocab:
                pred += f * W[w]
            # print(str(w) + " : " + str(f) + " : " + str(pred))
    if pred <= 0:
        return -1
    else:
        return 1


def read_model(model_file):
    parsed_file = json.load(open(model_file))
    w_td = parsed_file[0]
    w_pn = parsed_file[1]
    bias = parsed_file[2]
    vocabulary = parsed_file[3]
    # print(json.dumps(w_td, indent=2))
    # print(json.dumps(w_pn, indent=2))
    # print(bias)

    return w_td, w_pn, bias, vocabulary


if __name__ == '__main__':

    w_td, w_pn, bias, vocab = read_model(sys.argv[1])

    w_td = defaultdict(float, w_td)
    w_pn = defaultdict(float, w_pn)
    # print(w_td)
    # print(w_pn)
    vocab = defaultdict(int, vocab)
    all_files = glob.glob(os.path.join(sys.argv[2], '*/*/*/*.txt'))
    with open("percepoutput.txt", "w") as perceptron_output_file:
        for f in all_files:
            # class1, class2, fold, fname = f.split('\\')[-4:]
            class1, class2, fold, fname = f.split('/')[-4:]
            y_pred_td = classify(f, w_td, bias[0])
            y_pred_pn = classify(f, w_pn, bias[1])

            # write to file
            output_class = ""
            if y_pred_td == 1:
                output_class += "truthful "
            else:
                output_class += "deceptive "
            if y_pred_pn == 1:
                output_class += "positive "
            else:
                output_class += "negative "
            output_class += f + "\n"
            perceptron_output_file.write(output_class)
