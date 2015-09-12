__author__ = 'Jianxiang Fan'
__email__ = 'jianxiang.fan@colorado.edu'

import itertools
import argparse
import gzip


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

    :param target: Target words sequence
    :param source: Source words sequence
    """

    insert_cost = lambda s: 1
    delete_cost = lambda s: 1
    subst_cost = lambda s, t: 0 if s == t else 1
    n, m = len(target), len(source)

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


def word_error_rate(target, source):
    """
    Compute Word Error Rate (WER) of the target words sequence refering to the source one.
    The first step in computing WER is to compute the minimum number of edits. WER is then
    just the length normalized minimum edit distance (i.e, minimum edit distance divided
    by the length of the reference in terms of words).

    :param target: Target words sequence
    :param source: Source words sequence
    """
    return float(min_edit_dist(target, source)) / len(source)


def gzopen(filename, mode):
    """
    Open a gzip-compressed file or a plain text file.

    :param filename
    :param mode
    """
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
    argparser.add_argument("--refer", help="Reference file",
                           type=str, required=False)
    args = argparser.parse_args()

    clean_input = lambda s: s.decode(encoding='utf-8').strip(' \t\n\r#').lower() \
        if type(s) is not str else s.strip(' \t\n\r#').lower()

    with gzopen(args.lexicon, 'r') as f1, gzopen(args.target, 'r') as f2, open(args.output, 'w') as f3:
        lex = itertools.islice(
            (clean_input(line).split('\t')[0] for line in f1), args.limit)
        segmenter = Segmenter(lex)
        seg_answers = (segmenter.max_match(clean_input(line)) for line in f2)
        f4 = gzopen(args.refer, 'r') if args.refer else None
        ref_answers = (clean_input(line).split() for line in f4) if f4 else None
        wers = []
        for p in seg_answers:
            if ref_answers:
                wers.append(word_error_rate(p, next(ref_answers)))
            f3.write(' '.join(p) + '\n')
        if f4:
            f4.close()
        if wers:
            print(sum(wers) / len(wers))
