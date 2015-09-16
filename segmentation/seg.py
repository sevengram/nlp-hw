__author__ = 'Jianxiang Fan'
__email__ = 'jianxiang.fan@colorado.edu'

import argparse
import gzip

missing_words = ['mentalist', 'espy', 'ipad', 'cuboulder', 'iphone6s', 'teaman']


class Segmenter:
    def __init__(self, lexicon, split_words=None, extra_words=None, short_words=None):
        """
        Create segmenter using certain lexicon

        :param lexicon: The lexicon used to segement
        :param split_words: The dict to split words
        :param extra_words: Extra words should be added into lexicon
        :param short_words: Common short words list
        """
        self.lexicon = set(lexicon) | set(extra_words or [])
        self.word_max_len = max(len(w) for w in self.lexicon)
        self.split_words = split_words
        self.short_words = short_words

    def match(self, text, base=None):
        """
        Segment the text by MaxMatch use base algorithm and then split words

        :param text: Text to be segemented
        """
        if base == "back":
            match_func = self.back_max_match
        elif base == "frontback":
            match_func = self.front_back_max_match
        else:
            match_func = self.max_match
        result = match_func(text)

        # Recheck strange short words
        if self.short_words:
            for i in reversed(range(len(result))):
                if len(result[i]) <= 2 and result[i] not in self.short_words:
                    if i != 0:
                        result = self.back_max_match(''.join(result[:i + 1])) + result[i + 1:]
                        break
            for i in range(len(result)):
                if len(result[i]) <= 2 and result[i] not in self.short_words:
                    if i != len(result):
                        result = result[:i] + self.max_match(''.join(result[i:]))
                        break

        # Split some strange combine words
        if self.split_words:
            i = 0
            while i < len(result):
                token = result[i]
                if token in self.split_words:
                    words = self.split_words[token]
                    result[i:i + len(words) - 1] = words
                    i += len(words)
                else:
                    i += 1
        return result

    def front_back_max_match(self, text):
        """
        Segment the text by MaxMatch algorithm simultaneously from the front and back

        :param text: Text to be segemented
        """
        front_result, back_result = [], []
        l = len(text)
        fl, br = 0, l
        front_stop, back_stop = False, False
        while fl < br:
            if text[fl:br] in self.lexicon:
                front_result.append(text[fl:br])
                break
            fr = min(fl + self.word_max_len, br if back_stop else l)
            bl = max(br - self.word_max_len, fl if front_stop else 0)
            front_word = None
            back_word = None
            while not front_stop and fr > fl:
                if text[fl:fr] in self.lexicon:
                    front_word = text[fl:fr]
                    break
                else:
                    fr -= 1
            while not back_stop and br > bl:
                if text[bl:br] in self.lexicon:
                    back_word = text[bl:br]
                    break
                else:
                    bl += 1
            if front_word and back_word:
                if fr <= bl:
                    front_result.append(front_word)
                    back_result.append(back_word)
                    fl, br = fr, bl
                else:
                    if len(front_word) >= len(back_word):
                        back_stop = True
                        front_result.append(front_word)
                        fl = fr
                        while fr > br and back_result:
                            br += len(back_result.pop())
                    else:
                        front_stop = True
                        back_result.append(back_word)
                        br = bl
                        while bl < fl and front_result:
                            fl -= len(front_result.pop())
            elif front_word:
                front_result.append(front_word)
                fl = fr
                back_stop = True
            elif back_word:
                back_result.append(back_word)
                br = bl
                front_stop = True
            else:
                front_result.append(text[fl:br])
                fl = br
        return front_result + back_result[::-1]

    def back_max_match(self, text):
        """
        Segment the text by MaxMatch algorithm from the back

        :param text: Text to be segemented
        """
        result = []
        l = len(text)
        right = l
        while right > 0:
            left = max(right - self.word_max_len, 0)
            while right > left:
                if text[left:right] in self.lexicon:
                    result.append(text[left:right])
                    break
                else:
                    left += 1
            if right == left:
                result.append(text[right - 1])
                right -= 1
            else:
                right = left
        return result[::-1]

    def max_match(self, text):
        """
        Segment the text by MaxMatch algorithm from the front

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


def split_tokens(lexicon, refer_words, low_limit):
    """
    Split tokens into common words, return split result dict

    :param lexicon: The lexicon used to segement the token
    :param refer_words: Refer words used to split wokens
    :param low_limit: Split tokens after this threshold
    """
    l = list(lexicon)
    result = {}
    seg = Segmenter(refer_words)
    for i in range(low_limit, len(l)):
        split_result = seg.max_match(l[i])
        if len(split_result) == 2 and min(len(x) for x in split_result) > 1:
            result[l[i]] = split_result
    return result


def common_short_words(lexicon, word_length, bound, extra=None):
    return [w for w in lexicon[:bound] if 1 < len(w) <= word_length] + (extra or ['a', 'i', 'u'])


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--target", help="Target file to segment",
                           type=str, default="data/hashtags-dev.txt.gz", required=False)
    argparser.add_argument("--output", help="Output file",
                           type=str, default="out.txt", required=False)
    argparser.add_argument("--lexicon", help="Lexicon file",
                           type=str, default="data/bigwordlist.txt.gz", required=False)
    argparser.add_argument("--limit", help="Limit size of the lexicon",
                           type=int, default=75000, required=False)
    argparser.add_argument("--refer", help="Reference file",
                           type=str, required=False)
    argparser.add_argument("--bk", help="Use back max match", action='store_true')
    argparser.add_argument("--fb", help="Use frontback max match", action='store_true')
    argparser.add_argument("--ew", help="Use extra words", action='store_true')
    argparser.add_argument("--sw", help="Short word check", action='store_true')
    argparser.add_argument("--st", help="Split combine tokens", action='store_true')
    args = argparser.parse_args()

    clean_input = lambda s: s.decode(encoding='utf-8').strip(' \t\n\r#').lower() \
        if type(s) is not str else s.strip(' \t\n\r#').lower()

    with gzopen(args.lexicon, 'r') as f1, gzopen(args.target, 'r') as f2, open(args.output, 'w') as f3:
        lex = [clean_input(line).split('\t')[0] for line in f1][:args.limit]
        segmenter = Segmenter(lex,
                              split_words=split_tokens(lex, lex[:250], 20000) if args.st else None,
                              extra_words=missing_words if args.ew else None,
                              short_words=common_short_words(lex, 2, 200) if args.sw else None)

        base_match = "frontback" if args.fb else "back" if args.bk else "front"
        seg_answers = (segmenter.match(clean_input(line), base_match) for line in f2)
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
