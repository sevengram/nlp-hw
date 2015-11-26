__author__ = 'Jianxiang Fan'
__email__ = 'jianxiang.fan@colorado.edu'

import time
import argparse
from sklearn.cross_validation import ShuffleSplit, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier


def load_dataset(path):
    with open(path) as f:
        return [line.split('\t')[1] for line in f]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--true', metavar='FILE', help='true data file', default='./data/hotelT-train.txt', type=str)
    parser.add_argument('--false', metavar='FILE', help='false data file', default='./data/hotelF-train.txt', type=str)
    args = parser.parse_args()

    train_true = load_dataset(args.true)
    train_false = load_dataset(args.false)
    train = train_true + train_false
    y_train = [1] * len(train_true) + [0] * len(train_false)

    vectorizer = CountVectorizer(ngram_range=(1, 1), stop_words='english')
    x_train = vectorizer.fit_transform(train)
    # Train classifier and do cross validation
    lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
    print("Start cross validation...")
    cv = ShuffleSplit(len(y_train), 100, test_size=0.1, random_state=int(time.time()))
    scores = cross_val_score(lr, x_train, y_train, cv=cv, scoring='accuracy', verbose=1)
    print(scores)
    print(scores.mean(), scores.std())
