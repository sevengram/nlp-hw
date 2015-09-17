## Homework 1 - Segmentation

*Author: Jianxiang Fan*  
*Email: jianxiang.fan@colorado.edu*

### Usage
```
$ python seg.py --help
usage: seg.py [-h] [--target TARGET] [--output OUTPUT] [--lexicon LEXICON]
              [--limit LIMIT] [--refer REFER] [--bk] [--fb] [--ew] [--sw]
              [--st]

optional arguments:
  -h, --help         show this help message and exit
  --target TARGET    Target file to segment
  --output OUTPUT    Output file
  --lexicon LEXICON  Lexicon file
  --limit LIMIT      Limit size of the lexicon
  --refer REFER      Reference file
  --ew               Add extra words
  --bk               Use back max match
  --fb               Use frontback max match
  --sw               Short word check
  --st               Split combine tokens
```

### Original lexicon
First 75,000 most frequent words from a Google-derived list 

### Default MaxMatch
```
$ python seg.py --target=target_file --out=out_put_file --lexicon=lexicon_file
```
WER of the dev set: 0.661224489796


### Missing words in Lexicon (--ew)
Sometimes the segmenter fails just because it misses some words appearing in the golden answers, such as 'iphone6s', 'cuboulder' etc. In order to show the **real** performance of the **algorithm and strategy**, I added these missing words when working on the dev set.

WER of the dev set: 0.464795918367

### Back MaxMatch (--bk)
Start MaxMatch algorithm from the end of the string.
```
$ python seg.py --target=target_file --out=out_put_file --lexicon=lexicon_file --ew --bk
```
WER of the dev set: 0.274362244897

### Front Back MaxMatch (--fb)
Use MaxMatch algorithm from the start and the end of a string at the same time. It should deal with the overlapped words. 

*Pseudocode*
```
def front_back_max_match(text, lexicon):
    front_result = back_result = []
    front_stop = back_stop = False
    front_left = 0
    back_right = len(text)
    while front_left < back_right:
        if (text[front_left:back_right] in lexicon):
            add text[front_left:back_right] to front_result
            break
        if (not front_stop):
            do front-max-match(left_bound = front_left, right_bound = back_stop ? back_right : len(text))
        if (not back_stop):
            do back-max-match(left_bound = front_stop ? front_left : 0, right_bound = back_right)
        front_word = back_word = ''
        if (find both front_word and back_word and two words don't overlap):
            add front_word to front_result
            add back_word to back_result
            update front_left and back_right
        else if (find front_word or back_word or two words overlap):
            if (front_word is longer than back_word):
                add front_word to front_result
                update front_left
                back_stop = True
                if (front_word overlaps with the words already in back_result):
                    pop all these words from back_result
                    update back_right
            else:
                add back_word to back_result
                update back_right
                front_stop = True
                if (back_word overlaps with the words already in front_result):
                    pop all these words from front_result
                    update front_left
        else:
            add text[front_left:back_right] to front_result
            break
    return (properly concat front_result and back_result)
```
```
$ python seg.py --target=target_file --out=out_put_file --lexicon=lexicon_file --ew --fb
```
WER of the dev set: 0.170833333333

### Short words recheck (--sw)
Strange short words (length <= 2) are often the idicator of the segmentation error, such as 'yous noo ze you lose', 'there al hiphop'. 
I pick up a bunch of (default 30) common short words from the lexicon. If the result contains any short words not in this list, it will be rechecked.

*Common short words*
> ['of', 'to', 'in', 'is', 'on', 'by', 'it', 'or', 'be', 'at', 'as', 'an', 'we', 'us', 'if', 'my', 'do', 'no', 'he', 'up', 'so', 'pm', 'am', 'me', 're', 'go', 'de', 'a', 'i', 'u']

*Pseudocode*
```
def recheck_short_words(result):
    for word in reverse(result):
        if (word is short but not in common list):
            s = concat all the previous words in the result and this word
            result = back_max_match(s) + following words
    for word in result:
        if (word is short but not in common list):
            s = concat this word and all the following words in the result
            result = previous words + max_match(s)
    return result
```
```
$ python seg.py --target=target_file --out=out_put_file --lexicon=lexicon_file --ew --fb --sw
```
WER of the dev set: 0.145535714286

### Split combined words (--st)
There are some uncommon **combined** words in our lexion, such as 'ofthe', 'tobe', 'forthe', which will be figured out by MaxMatch algorithm and block those very common word pairs - 'of the', 'to be', 'for the'. Split these words are useful to improve the whole accuracy in pratice.

*Combined words*
> {'ifas': ['if', 'as'], 'allis': ['all', 'is'], 'andthe': ['and', 'the'], 'oris': ['or', 'is'], 'nosearch': ['no', 'search'], 'tobe': ['to', 'be'], 'newby': ['new', 'by'], 'inone': ['in', 'one'], 'anand': ['an', 'and'], 'orin': ['or', 'in'], 'canis': ['can', 'is'], 'inno': ['in', 'no'], 'ordo': ['or', 'do'], 'fromthe': ['from', 'the'], 'allin': ['all', 'in'], 'onus': ['on', 'us'], 'canto': ['can', 'to'], 'asif': ['as', 'if'], 'toto': ['to', 'to'], 'tothe': ['to', 'the'], 'atto': ['at', 'to'], 'anno': ['an', 'no'], 'forthe': ['for', 'the'], 'atis': ['at', 'is'], 'andnot': ['and', 'not'], 'noor': ['no', 'or'], 'beit': ['be', 'it'], 'aboutus': ['about', 'us'], 'inthe': ['in', 'the'], 'asis': ['as', 'is'], 'bein': ['be', 'in'], 'canmore': ['can', 'more'], 'anus': ['an', 'us'], 'ofthe': ['of', 'the'], 'usno': ['us', 'no'], 'wean': ['we', 'an'], 'doin': ['do', 'in'], 'weis': ['we', 'is'], 'doit': ['do', 'it'], 'wein': ['we', 'in'], 'doan': ['do', 'an'], 'weare': ['we', 'are'], 'itat': ['it', 'at'], 'dois': ['do', 'is'], 'doon': ['do', 'on'], 'bythe': ['by', 'the'], 'infor': ['in', 'for'], 'isas': ['is', 'as'], 'onthe': ['on', 'the'], 'bebe': ['be', 'be'], 'tomy': ['to', 'my'], 'anat': ['an', 'at'], 'theor': ['the', 'or'], 'onetime': ['one', 'time'], 'anas': ['an', 'as'], 'befor': ['be', 'for'], 'nono': ['no', 'no'], 'canby': ['can', 'by'], 'noone': ['no', 'one'], 'dodo': ['do', 'do'], 'forno': ['for', 'no'], 'hasan': ['has', 'an'], 'onan': ['on', 'an'], 'moreno': ['more', 'no'], 'toit': ['to', 'it'], 'nowe': ['no', 'we'], 'forall': ['for', 'all'], 'isin': ['is', 'in'], 'otherother': ['other', 'other'], 'itis': ['it', 'is'], 'homehome': ['home', 'home'], 'oran': ['or', 'an'], 'beall': ['be', 'all']}

```
$ python seg.py --target=target_file --out=out_put_file --lexicon=lexicon_file --ew --fb --sw --st
```
WER of the dev set: 0.105654761905