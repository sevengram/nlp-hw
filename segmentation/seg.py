__author__ = 'Jianxiang Fan'
__email__ = 'jianxiang.fan@colorado.edu'

import itertools
import argparse
import gzip
import time


class Segmenter:
    def __init__(self, lexicon):
        """
        Create segmenter using certain lexicon

        :param lexicon: The lexicon used to segement
        """
        self.lexicon = set(lexicon)
        self.word_max_len = max(len(w) for w in self.lexicon)

    def max_match(self, text):
        """
        Segment the text by MaxMatch algorithm

        :param text: Text to be segemented
        """
        result = []
        l = len(text)
        left = 0
        while left < l:
            right = min(left + self.word_max_len, l)
            while right > left:
                if text[left:right] in self.lexicon:
                    result.append(text[left:right])
                    break
                else:
                    right -= 1
            if right == left:
                result.append(text[left])
                left += 1
            else:
                left = right
        return result


def min_edit_dist(target, source):
    """
    Computes the min edit distance from target to source. Figure 3.25 in the book. Assume that
    insertions, deletions and (actual) substitutions all cost 1 for this HW. Note the indexes are a
    little different from the text. There we are assuming the source and target indexing starts a 1.
    Here we are using 0-based indexing.

    :param target: Target words
    :param source: Source words
    """

    insert_cost = lambda s: 1
    delete_cost = lambda s: 1
    subst_cost = lambda s, t: 0 if s == t else 1

    n = len(target)
    m = len(source)

    distance = [[0 for i in range(m + 1)] for j in range(n + 1)]

    for i in range(1, n + 1):
        distance[i][0] = distance[i - 1][0] + insert_cost(target[i - 1])

    for j in range(1, m + 1):
        distance[0][j] = distance[0][j - 1] + delete_cost(source[j - 1])

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            distance[i][j] = min(distance[i - 1][j] + insert_cost(target[i - 1]),
                                 distance[i][j - 1] + insert_cost(source[j - 1]),
                                 distance[i - 1][j - 1] + subst_cost(source[j - 1], target[i - 1]))
    return distance[n][m]


def gzopen(filename, mode):
    if filename.endswith('gz'):
        return gzip.open(filename, mode)
    else:
        return open(filename, mode)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--lexicon", help="Lexicon file",
                           type=str, default="data/bigwordlist.txt.gz", required=False)
    argparser.add_argument("--limit", help="Limit of the words in lexicon",
                           type=int, default=75000, required=False)
    argparser.add_argument("--target", help="Target file",
                           type=str, default="data/hashtags-dev.txt.gz", required=False)
    argparser.add_argument("--output", help="Output file",
                           type=str, default="out.txt", required=False)

    args = argparser.parse_args()

    t1 = time.time()
    with gzopen(args.lexicon, 'r') as f1, gzopen(args.target, 'r') as f2, open(args.output, 'w') as f3:
        lex = itertools.islice(
            (line.decode(encoding='utf-8').strip().split('\t')[0].lower() for line in f1), args.limit)
        segmenter = Segmenter(lex)
        for p in (segmenter.max_match(line.decode(encoding='utf-8').strip(' \t\n\r#').lower()) for line in f2):
            f3.write(' '.join(p) + '\n')
        print(time.time() - t1)
