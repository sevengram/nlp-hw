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
                s = text[left:right]
                if s in self.lexicon:
                    result.append(s)
                    left = right - 1
                    break
                else:
                    right -= 1
            left += 1
        return result


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

    with gzip.open(args.lexicon, 'r') as f1, gzip.open(args.target, 'r') as f2, open(args.output, 'w') as f3:
        lex = itertools.islice(
            (line.decode(encoding='utf-8').strip().split('\t')[0].lower() for line in f1), args.limit)
        segmenter = Segmenter(lex)
        t1 = time.time()
        for p in (segmenter.max_match(line.decode(encoding='utf-8').strip(' \t\n\r#').lower()) for line in f2):
            f3.write(' '.join(p) + '\n')
        print(time.time() - t1)
