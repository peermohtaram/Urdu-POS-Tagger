import collections
import itertools
import nltk
import math
import time
import sys
import ast
import numpy as np
from collections import defaultdict, deque

START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
RARE_SYMBOL = '_RARE_'
RARE_WORD_MAX_FREQ = 1
LOG_PROB_OF_ZERO = -1000

#This function takes the input `training` data, make tuple of word and tag and the senteces based on full stop mark (.).

#**Output:** `[('اونٹ', 'NN'), ('کی', 'PSP'), ('سفاری', 'NN'), ('خاص', 'JJ'), ('پسند', 'NN'), ('ہو', 'VBF'), ('سکتی', 'AUXM'), ('ہے', 'AUXT'), ('۔', 'PU')]`
def data_convert(train_file):
  data = ""

  for line in train_file:
      line = line.split("\n")[0]
      line = tuple(line.split("\t"))
      data+= str(line)+", "
  data =  data.replace("('۔', 'PU')", "('۔', 'PU')__")
  data_converted = ["["+ i.lstrip(", ")+"]" for i in data.split("__")][:-1]
  return data_converted

#This function takes the training data and separates the words and tags and adds them to separate lists.
#Each sentence is tokenized into its words and tags and appended into a list.
#**Sentence Format:** ```[['*', '*', 'w1', 'w2', 'STOP'], ['*', '*', 'w1', 'w2', 'STOP']]```
#**Tags Format:** `[['*', '*', 't1', 't2', 'STOP'], ['*', '*', 't1', 't2', 'STOP']]`
#Each sub-list shows a complete sentence. Double '*' and 'STOP' symbols are added in the start and end of a sentence respectively.
def wordstags_split(in_file):
  temp_list = in_file
  temp_sent = []
  temp_tags = []
  for each in temp_list:
    sentc, tags = zip(*ast.literal_eval(each))
    temp_sent.append([START_SYMBOL] * 2 + list(sentc) + [STOP_SYMBOL])
    temp_tags.append([START_SYMBOL] *2 + list(tags) + [STOP_SYMBOL])
  return temp_sent, temp_tags

#This function takes the `test data` as input and make senteces of test data, with each sentence on a single line.
def make_test(test_file):
  data = ""

  for line in test_file:
    line = line.split("\n")[0]
    data+= " "+str(line)

  data =  data.replace("۔ ", "۔__").split('__')
  data_converted = [[w.strip() for w in each.split()] for each in data]
  return data_converted

#This function takes the tags list and make trigrams.
def calc_trigrams(urdu_tags):
    bigram_c = collections.defaultdict(int)
    trigram_c = collections.defaultdict(int)

    for sentence in urdu_tags:
        for bigram in nltk.bigrams(sentence):
            bigram_c[bigram] += 1

    for sentence in urdu_tags:
        for trigram in nltk.trigrams(sentence):
            trigram_c[trigram] += 1

    return bigram_c, trigram_c

#This function implements the Kneser-Ney smoothing using built-in function in NLTK.
def kneser_ney(tri_grams):
  freq_dist = nltk.probability.FreqDist([*tri_grams])
  for k, v in freq_dist.items():
    freq_dist[k] = tri_grams[k]
  KN = nltk.KneserNeyProbDist(freq_dist)
  KNDict = {}
  for i in KN.samples():
      KNDict[i] = KN.prob(i)
  return KNDict

#This function takes the words from the training data and returns a python list of all of the words that occur more than value of `RARE_SYMBOL` parameter.
def calc_known(urdu_words):
    known_words = set()
    word_c = defaultdict(int)

    for sent_words in urdu_words:
        for word in sent_words:
            word_c[word] += 1

    for word, count in word_c.items():
        if count > RARE_WORD_MAX_FREQ:
            known_words.add(word)
    return known_words

#This function takes a set of sentences and a set of words that should not be marked `_RARE_`. Outputs a version of the set of sentences with rare words marked `_RARE_`
def replace_rare(urdu_words, known_words):
    for i, sent_words in enumerate(urdu_words):
        for j, word in enumerate(sent_words):
            if word not in known_words:
                urdu_words[i][j] = RARE_SYMBOL
    return urdu_words

#This function calculates emission probabilities and creates a list of possible tags.
#The first return value is a python dictionary where each key is a tuple in which the first element is a word and the second is a tag and the value is the log probability of that word/tag pair and the second return value is a list of possible tags for this data set
def calc_emission(urdu_words_rare, urdu_tags):
    e_values = {}
    e_values_c = collections.defaultdict(int)
    tags_c = collections.defaultdict(int)

    for word_sentence, tag_sentence in zip(urdu_words_rare, urdu_tags):
        for word, tag in zip(word_sentence, tag_sentence):
            e_values_c[(word, tag)] += 1
            tags_c[tag] += 1

    for (word, tag), p in e_values_c.items():
        e_values[(word, tag)] = math.log(float(p) / tags_c[tag], 2)

    return e_values, set(tags_c)

#This function takes data to tag , possible tags (taglist), a list of known words (knownwords), trigram probabilities (qvalues) and emission probabilities (evalues) and outputs a list where every element is a string of a sentence tagged in the `WORD   TAG` format
#`qvalues` is from the return of calc_trigrams = probability of the trigrams of tags
#`evalues` is from the return of calc_emission()
#Tagged is a list of tagged sentences in the format `WORD  TAG`. Each sentence is a string with a terminal newline, not a list of tokens.
def viterbi(urdu_dev_words, taglist, known_words, q_values, e_values):
    tagged = []
    pi = collections.defaultdict(float)
    bp = {}
    bp[(-1, START_SYMBOL, START_SYMBOL)] = START_SYMBOL
    pi[(-1, START_SYMBOL, START_SYMBOL)] = 0.0

    for tokens_orig in urdu_dev_words:
        tokens = [w if w in known_words else RARE_SYMBOL for w in tokens_orig]

        # k = 1 case
        for w in taglist:
            word_tag = (tokens[0], w)
            trigram = (START_SYMBOL, START_SYMBOL, w)
            pi[(0, START_SYMBOL, w)] = pi[(-1, START_SYMBOL, START_SYMBOL)] + q_values.get(trigram, LOG_PROB_OF_ZERO) + e_values.get(word_tag, LOG_PROB_OF_ZERO)
            bp[(0, START_SYMBOL, w)] = START_SYMBOL

        # k = 2 case
        for w in taglist:
            for u in taglist:
                word_tag = (tokens[1], u)
                trigram = (START_SYMBOL, w, u)
                pi[(1, w, u)] = pi.get((0, START_SYMBOL, w), LOG_PROB_OF_ZERO) + q_values.get(trigram, LOG_PROB_OF_ZERO) + e_values.get(word_tag, LOG_PROB_OF_ZERO)
                bp[(1, w, u)] = START_SYMBOL

        # k >= 2 case
        for k in range(2, len(tokens)):
            for u in taglist:
                for v in taglist:
                    max_prob = float('-Inf')
                    max_tag = ''
                    for w in taglist:
                        score = pi.get((k - 1, w, u), LOG_PROB_OF_ZERO) + q_values.get((w, u, v), LOG_PROB_OF_ZERO) + e_values.get((tokens[k], v), LOG_PROB_OF_ZERO)
                        if (score > max_prob):
                            max_prob = score
                            max_tag = w
                    bp[(k, u, v)] = max_tag
                    pi[(k, u, v)] = max_prob

        max_prob = float('-Inf')
        v_max, u_max = None, None
        # finding the max probability of last two tags
        for (u, v) in itertools.product(taglist, taglist):
            score = pi.get((len(tokens_orig) - 1, u, v), LOG_PROB_OF_ZERO) + q_values.get((u, v, STOP_SYMBOL), LOG_PROB_OF_ZERO)
            if score > max_prob:
                max_prob = score
                u_max = u
                v_max = v
        # append tags in reverse order
        tags = []
        tags.append(v_max)
        tags.append(u_max)

        for count, k in enumerate(range(len(tokens_orig) - 3, -1, -1)):
            tags.append(bp[(k + 2, tags[count + 1], tags[count])])

        tagged_sentence = ""
        # reverse tags
        tags.reverse()
        # stringify tags paired with word without start and stop symbols
        for k in range(0, len(tokens_orig)):
            tagged.append(tagged_sentence + tokens_orig[k] + "\t" + str(tags[k]))

    return tagged

#This function writes the output of `viterbi` (tagged data) into a text file and save on the current/specified location.
def q5_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
        outfile.write('\n')
    outfile.close()
  
def main():

    time.clock()

    infile = open(sys.argv[1], 'r', encoding='utf-8-sig')
    train = infile.readlines()
    infile.close()

    test_file = open(sys.argv[2], 'r', encoding='utf-8-sig')
    test = test_file.readlines()
    infile.close()

    train_data = data_convert(train)
    urdu_words, urdu_tags = wordstags_split(train_data)
    test_words = make_test(test)
    print('Train and Test Files Read')
    print('Time: ' + str(time.clock()) + ' sec')

    bigram_c, trigram_c = calc_trigrams(urdu_tags)
    q_values = kneser_ney(trigram_c)
    print('Smoothing Applied')
    print('Time: ' + str(time.clock()) + ' sec')

    known_words = calc_known(urdu_words)
    urdu_words_rare = replace_rare(urdu_words, known_words)

    e_values, taglist = calc_emission(urdu_words_rare, urdu_tags)
    print('e values computed')

    # del urdu_train
    # del urdu_words_rare

    print('Viterbi Starting')
    print('Time: ' + str(time.clock()) + ' sec')

    viterbi_tagged = viterbi(test_words, taglist, known_words, q_values, e_values)

    print('Viterbi Done')
    print('Time: ' + str(time.clock()) + ' sec')

    q5_output(viterbi_tagged, 'tagged_output.txt')

    print('Time: ' + str(time.clock()) + ' sec')

if __name__=='__main__':
    main()