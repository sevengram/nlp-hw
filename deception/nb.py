__author__ = 'Jianxiang Fan'
__email__ = 'jianxiang.fan@colorado.edu'

import math
import string
import argparse
import random

import numpy


def base_filter():
    return string.punctuation + '\t\n\r'


def text_to_sequence(text, filters=base_filter(), split=" "):
    text = text.lower().translate(string.maketrans(filters, split * len(filters)))
    seq = text.split(split)
    return [_f for _f in seq if _f]


def load_dataset(path, label=None, k=0):
    d = []
    with open(path) as f:
        for line in f:
            if line:
                p = line.split('\t')
                d.append((p[0], p[1], label))
    if k != 0:
        random.shuffle(d)
        dev_size = len(d) / k
        for i in range(k):
            yield d[:i * dev_size] + d[(i + 1) * dev_size:], d[i * dev_size:(i + 1) * dev_size]
    else:
        yield d


class Classifier(object):
    def __init__(self, top_trunk=20, lower_bound=1):
        self.doc_counts = {}
        self.word_counts = {}
        self.true_counts = {}
        self.false_counts = {}
        self.prior = {}
        self.true_likelihood = {}
        self.false_likelihood = {}
        self.effective_words = None
        self.lower_bound = lower_bound
        self.top_trunk = top_trunk

    def clear(self):
        self.doc_counts.clear()
        self.word_counts.clear()
        self.true_counts.clear()
        self.false_counts.clear()

    def fit_on_texts(self, data):
        for tid, text, label in data:
            self.doc_counts[label] = self.doc_counts.get(label, 0) + 1
            for w in text_to_sequence(text):
                self.word_counts[w] = self.word_counts.get(w, 0) + 1
                if label == 1:
                    self.true_counts[w] = self.true_counts.get(w, 0) + 1
                else:
                    self.false_counts[w] = self.false_counts.get(w, 0) + 1

    def compute_likelihood(self):
        self.effective_words = [k for k, v in self.word_counts.iteritems() if
                                v > 1 and len(k) > 1 and not k[0].isdigit() and k not in stop_words]
        vocab_size = len(self.effective_words)
        for w in self.effective_words:
            self.true_likelihood[w] = math.log(
                float(self.true_counts.get(w, 0) + 1) / (self.word_counts.get(w, 0) + vocab_size))
            self.false_likelihood[w] = math.log(
                float(self.false_counts.get(w, 0) + 1) / (self.word_counts.get(w, 0) + vocab_size))

    def classify(self, text):
        true_prob = self.prior[1]
        false_prob = self.prior[0]
        for w in text_to_sequence(text):
            true_prob += self.true_likelihood.get(w, 0)
            false_prob += self.false_likelihood.get(w, 0)
        return 1 if true_prob >= false_prob else 0

    def compute_prior(self):
        self.prior[0] = math.log(float(self.doc_counts[0]) / (self.doc_counts[0] + self.doc_counts[1]))
        self.prior[1] = math.log(float(self.doc_counts[1]) / (self.doc_counts[0] + self.doc_counts[1]))


stop_words = frozenset([
    "a", "about", "above", "across", "after", "afterwards", "again", "against",
    "all", "almost", "alone", "along", "already", "also", "although", "always",
    "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
    "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
    "around", "as", "at", "back", "be", "became", "because", "become",
    "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
    "below", "beside", "besides", "between", "beyond", "bill", "both",
    "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con",
    "could", "couldnt", "cry", "de", "describe", "detail", "do", "done",
    "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else",
    "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
    "everything", "everywhere", "except", "few", "fifteen", "fify", "fill",
    "find", "fire", "first", "five", "for", "former", "formerly", "forty",
    "found", "four", "from", "front", "full", "further", "get", "give", "go",
    "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
    "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
    "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed",
    "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter",
    "latterly", "least", "less", "ltd", "made", "many", "may", "me",
    "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly",
    "move", "much", "must", "my", "myself", "name", "namely", "neither",
    "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone",
    "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on",
    "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our",
    "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
    "please", "put", "rather", "re", "same", "see", "seem", "seemed",
    "seeming", "seems", "serious", "several", "she", "should", "show", "side",
    "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone",
    "something", "sometime", "sometimes", "somewhere", "still", "such",
    "system", "take", "ten", "than", "that", "the", "their", "them",
    "themselves", "then", "thence", "there", "thereafter", "thereby",
    "therefore", "therein", "thereupon", "these", "they", "thick", "thin",
    "third", "this", "those", "though", "three", "through", "throughout",
    "thru", "thus", "to", "together", "too", "top", "toward", "towards",
    "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us",
    "very", "via", "was", "we", "well", "were", "what", "whatever", "when",
    "whence", "whenever", "where", "whereafter", "whereas", "whereby",
    "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
    "who", "whoever", "whole", "whom", "whose", "why", "will", "with",
    "within", "without", "would", "yet", "you", "your", "yours", "yourself",
    "yourselves"])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--true', metavar='FILE', help='true data file', default='./data/hotelT-train.txt', type=str)
    parser.add_argument('--false', metavar='FILE', help='false data file', default='./data/hotelF-train.txt', type=str)
    parser.add_argument('--test', metavar='FILE', help='test data file', type=str)
    parser.add_argument('--k', metavar='SIZE', help='k-fold size', default=0, type=int, required=False)
    args = parser.parse_args()

    classifier = Classifier()
    true_set = load_dataset(args.true, label=1, k=args.k)
    false_set = load_dataset(args.false, label=0, k=args.k)
    error_count, test_count = 0, 0
    if args.k != 0:
        result = []
        for ii in range(args.k):
            classifier.clear()
            true_train_set, true_dev_set = true_set.next()
            false_train_set, false_dev_set = false_set.next()
            classifier.fit_on_texts(true_train_set)
            classifier.fit_on_texts(false_train_set)
            classifier.compute_prior()
            classifier.compute_likelihood()
            test_count, error_count = 0, 0
            test_count = len(true_dev_set) + len(false_dev_set)
            for mid, t, l in true_dev_set:
                error_count += 1 - classifier.classify(t)
            for mid, t, l in false_dev_set:
                error_count += classifier.classify(t)
            result.append(1 - float(error_count) / test_count)
        print(numpy.mean(result), numpy.var(result))
    else:
        classifier.fit_on_texts(true_set.next())
        classifier.fit_on_texts(false_set.next())
        classifier.compute_prior()
        classifier.compute_likelihood()
        if args.test is not None:
            test_set = load_dataset(args.test).next()
            for mid, t, l in test_set:
                print('%s\t%s' % (mid, 'T' if classifier.classify(t) == 1 else 'F'))
