__author__ = 'Jianxiang Fan'
__email__ = 'jianxiang.fan@colorado.edu'

import math
import string
import argparse
import random

from stop_words import ENGLISH_STOP_WORDS


def base_filter():
    return string.punctuation + '\t\n\r'


def text_to_word_sequence(text, filters=base_filter(), split=" "):
    text = text.lower().translate(string.maketrans(filters, split * len(filters)))
    seq = text.split(split)
    return [_f for _f in seq if _f]


def load_dataset(path, label, k):
    with open(path) as f:
        d = [(line.split('\t')[1], label) for line in f]
    random.shuffle(d)
    dev_size = len(d) / k
    for i in range(k):
        yield d[:i * dev_size] + d[(i + 1) * dev_size:], d[i * dev_size:(i + 1) * dev_size]


class Classifier(object):
    def __init__(self, top_trunk=20, lower_bound=1):
        self.doc_counts = {}
        self.word_counts = {}
        self.pos_counts = {}
        self.neg_counts = {}
        self.prior = {}
        self.pos_likelihood = {}
        self.neg_likelihood = {}
        self.effective_words = None
        self.lower_bound = lower_bound
        self.top_trunk = top_trunk

    def clear(self):
        self.doc_counts.clear()
        self.word_counts.clear()
        self.pos_counts.clear()
        self.neg_counts.clear()

    def fit_on_texts(self, data):
        for text, label in data:
            self.doc_counts[label] = self.doc_counts.get(label, 0) + 1
            for w in text_to_word_sequence(text):
                self.word_counts[w] = self.word_counts.get(w, 0) + 1
                if label == 1:
                    self.pos_counts[w] = self.pos_counts.get(w, 0) + 1
                else:
                    self.neg_counts[w] = self.neg_counts.get(w, 0) + 1

    def compute_likelihood(self):
        self.effective_words = [k for k, v in self.word_counts.iteritems() if
                                v > 1 and len(k) > 1 and not k[0].isdigit() and k not in ENGLISH_STOP_WORDS]
        vocab_size = len(self.effective_words)
        for w in self.effective_words:
            self.pos_likelihood[w] = math.log(
                float(self.pos_counts.get(w, 0) + 1) / (self.word_counts.get(w, 0) + vocab_size))
            self.neg_likelihood[w] = math.log(
                float(self.neg_counts.get(w, 0) + 1) / (self.word_counts.get(w, 0) + vocab_size))

    def classify(self, text):
        pos_prob = self.prior[1]
        neg_prob = self.prior[0]
        for w in text_to_word_sequence(text):
            pos_prob += self.pos_likelihood.get(w, 0)
            neg_prob += self.neg_likelihood.get(w, 0)
        return 1 if pos_prob >= neg_prob else 0

    def compute_prior(self):
        self.prior[0] = math.log(float(self.doc_counts[0]) / (self.doc_counts[0] + self.doc_counts[1]))
        self.prior[1] = math.log(float(self.doc_counts[1]) / (self.doc_counts[0] + self.doc_counts[1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pos', metavar='FILE', help='positive data file', type=str, required=True)
    parser.add_argument('--neg', metavar='FILE', help='negative data file', type=str, required=True)
    parser.add_argument('--k', metavar='SIZE', help='k-fold size', default=5, type=int, required=False)
    args = parser.parse_args()

    classifier = Classifier()
    pos_set = load_dataset(args.pos, label=1, k=args.k)
    neg_set = load_dataset(args.neg, label=0, k=args.k)
    error_count, test_count = 0, 0
    for ii in range(args.k):
        classifier.clear()
        pos_train_set, pos_dev_set = pos_set.next()
        neg_train_set, neg_dev_set = neg_set.next()
        classifier.fit_on_texts(pos_train_set)
        classifier.fit_on_texts(neg_train_set)
        classifier.compute_prior()
        classifier.compute_likelihood()
        test_count += len(pos_dev_set) + len(neg_dev_set)
        for t, l in pos_dev_set:
            error_count += 1 - classifier.classify(t)
        for t, l in neg_dev_set:
            error_count += classifier.classify(t)
    print(1 - float(error_count) / test_count)
