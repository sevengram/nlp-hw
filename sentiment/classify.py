__author__ = 'Jianxiang Fan'
__email__ = 'jianxiang.fan@colorado.edu'

import math
import string
import argparse
import random


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
        for tid, text, label in data:
            self.doc_counts[label] = self.doc_counts.get(label, 0) + 1
            for w in text_to_sequence(text):
                self.word_counts[w] = self.word_counts.get(w, 0) + 1
                if label == 1:
                    self.pos_counts[w] = self.pos_counts.get(w, 0) + 1
                else:
                    self.neg_counts[w] = self.neg_counts.get(w, 0) + 1

    def compute_likelihood(self):
        self.effective_words = [k for k, v in self.word_counts.iteritems() if
                                v > 1 and len(k) > 1 and not k[0].isdigit() and k not in stop_words]
        vocab_size = len(self.effective_words)
        for w in self.effective_words:
            self.pos_likelihood[w] = math.log(
                float(self.pos_counts.get(w, 0) + 1) / (self.word_counts.get(w, 0) + vocab_size))
            self.neg_likelihood[w] = math.log(
                float(self.neg_counts.get(w, 0) + 1) / (self.word_counts.get(w, 0) + vocab_size))

    def classify(self, text):
        pos_prob = self.prior[1]
        neg_prob = self.prior[0]
        for w in text_to_sequence(text):
            pos_prob += self.pos_likelihood.get(w, 0)
            neg_prob += self.neg_likelihood.get(w, 0)
        return 1 if pos_prob >= neg_prob else 0

    def compute_prior(self):
        self.prior[0] = math.log(float(self.doc_counts[0]) / (self.doc_counts[0] + self.doc_counts[1]))
        self.prior[1] = math.log(float(self.doc_counts[1]) / (self.doc_counts[0] + self.doc_counts[1]))


stop_words = frozenset([
    "a", "about", "above", "across", "after", "afterwards",
    "all", "almost", "alone", "along", "already", "also", "always",
    "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
    "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
    "around", "as", "at", "be", "became", "because", "become",
    "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
    "below", "beside", "besides", "between", "beyond", "both", "bottom",
    "by", "can", "cannot", "cant", "co", "con", "could", "couldnt", "de", "do", "done",
    "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else",
    "elsewhere", "etc", "even", "ever", "every", "everyone",
    "everything", "everywhere", "fifteen", "fifty", "fill",
    "find", "fire", "first", "for", "forty", "found", "from", "further", "get", "give", "go",
    "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
    "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
    "how", "hundred", "i", "ie", "if", "in", "inc", "indeed",
    "into", "is", "it", "its", "itself", "ny", "ltd", "made", "may", "me",
    "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly",
    "must", "my", "myself", "name", "namely", "nevertheless", "nine", "no", "none", "before",
    "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on",
    "once", "onto", "or", "other", "others", "otherwise", "our",
    "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
    "please", "put", "rather", "re", "same", "see", "seem", "seemed",
    "seeming", "seems", "serious", "several", "she", "should", "show", "side",
    "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone",
    "something", "sometime", "sometimes", "somewhere", "still", "such",
    "take", "ten", "than", "that", "the", "their", "them",
    "themselves", "then", "thence", "there", "thereafter", "thereby",
    "therefore", "therein", "thereupon", "these", "they",
    "third", "this", "those", "though", "through", "throughout",
    "thru", "thus", "to", "together", "too", "toward", "towards",
    "twelve", "twenty", "un", "under", "until", "up", "upon", "us",
    "very", "via", "was", "we", "were", "what", "whatever", "when",
    "whence", "whenever", "where", "whereafter", "whereas", "whereby",
    "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
    "who", "whoever", "whole", "whom", "whose", "why", "with",
    "within", "would", "yet", "you", "your", "yours", "yourself",
    "yourselves"])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pos', metavar='FILE', help='positive data file', type=str, required=True)
    parser.add_argument('--neg', metavar='FILE', help='negative data file', type=str, required=True)
    parser.add_argument('--test', metavar='FILE', help='test data file', type=str)
    parser.add_argument('--k', metavar='SIZE', help='k-fold size', default=0, type=int, required=False)
    args = parser.parse_args()

    classifier = Classifier()
    pos_set = load_dataset(args.pos, label=1, k=args.k)
    neg_set = load_dataset(args.neg, label=0, k=args.k)
    error_count, test_count = 0, 0
    if args.k != 0:
        for ii in range(args.k):
            classifier.clear()
            pos_train_set, pos_dev_set = pos_set.next()
            neg_train_set, neg_dev_set = neg_set.next()
            classifier.fit_on_texts(pos_train_set)
            classifier.fit_on_texts(neg_train_set)
            classifier.compute_prior()
            classifier.compute_likelihood()
            test_count += len(pos_dev_set) + len(neg_dev_set)
            for mid, t, l in pos_dev_set:
                error_count += 1 - classifier.classify(t)
            for mid, t, l in neg_dev_set:
                error_count += classifier.classify(t)
        print(1 - float(error_count) / test_count)
    else:
        classifier.fit_on_texts(pos_set.next())
        classifier.fit_on_texts(neg_set.next())
        classifier.compute_prior()
        classifier.compute_likelihood()
        if args.test is not None:
            test_set = load_dataset(args.test).next()
            for mid, t, l in test_set:
                print('%s\t%s' % (mid, 'POS' if classifier.classify(t) == 1 else 'NEG'))
