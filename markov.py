#!/usr/bin/env python

import random
import os
import sys
import argparse
import time
import glob
import fileinput
try: # cPickle is faster than pickle
    import cPickle as pickle
except ImportError:
    import pickle
from collections import defaultdict
import re

class Tokeniser:
    """Flexible tokeniser for the Markov chain.
    """

    def __init__(self, stream=None, noparagraphs=False):
        self.stream = sys.stdin if stream is None else stream
        self.orderoparagraphs = noparagraphs

    def __iter__(self):
        self.buffer = ''
        self.tok = ''
        self.halt = False
        return self

    def next(self):
        while not self.halt:
            # Return a pending token, if we have one
            if self.tok:
                out = self.tok
                self.tok = ''
                return out

            # Read the next character. If EOF, return what we have in the
            # buffer as the final token. Set a flag so we know to terminate
            # after this point.
            try:
                next_char = next(self.stream)
            except:
                next_char = ''
                self.halt = True
                if not self.buffer:
                    break

            # Determine if we have a new token
            out = None
            if self.buffer:
                punctuation = not next_char.isalnum() and not next_char.isspace()

                if self.buffer == '\n' and next_char == '\n':
                    # Paragraph break
                    if not self.orderoparagraphs:
                        out = self.buffer + next_char
                        next_char = ''

                elif not self.buffer.isspace() and (next_char.isspace() or punctuation):
                    # A word
                    out = self.buffer

                # If the next_char is a token, save it
                if punctuation:
                    self.tok = next_char
                    next_char = ''

            # If a token has been found, reset the buffer
            if out:
                self.buffer = ''

            # If the buffer is only spaces, clear it when a word is added
            if self.buffer.isspace() and not next_char.isspace():
                self.buffer = next_char
            else:
                self.buffer += next_char

            # Return the found token
            if out:
                return out

        # If we're here, we got nothing but EOF.
        raise StopIteration

###############################################################################
# Markov chain table
###############################################################################
def _newDefaultProb():
    return 1.0

def _newProbDict():
    return defaultdict(_newDefaultProb)

def _newTokenDict():
    return defaultdict(_newProbDict)

class Markov:
    CLAUSE_ENDS = [',', '.', ';', ':', '!', '?']
    SENTENCE_ENDS = ['!', '?', '.']

    def __init__(self, order=3):
        self.order = order
        self.seed = None
        self.prev = ('',)
        self.data = _newTokenDict()

    def train(self, tokenStream):
        # compile tokens into sentences
        print("Compiling sentences...")
        sentences = []
        sentence = []
        for token in tokenStream:
            sentence.append(token)
            if token in self.CLAUSE_ENDS: # if token is the end of a sentence
                sentences.append(sentence)
                sentence = []
        print("Compiled {count:d} sentences.".format(count=len(sentences)));

        # create a chain database from the sentences
        print("Building markov chains...")
        self.data[('',)][''] = 0.0 # default
        for words in sentences:
            if (len(words) == 0):
                continue

            self.data[('',)][words[0]] += 1 # first word in sentence

            for order in range(1, self.order + 1):
                for idx in range(len(words) - 1):
                    if (idx + order) >= len(words):
                        continue
                    word = tuple(words[idx:idx + order])
                    self.data[word][words[idx + order]] += 1
                # record last word as an end (skip the end punctuation)
                self.data[tuple(words[len(words) - order - 1:len(words) - 1])][''] += 1

        # normalize word counts -> probabilities (0 -> 1)
        print("Normalizing data...")
        for key in self.data:
            total = 0
            for chain in self.data[key]:
                total += self.data[key][chain]
            if total > 0:
                for chain in self.data[key]:
                    self.data[key][chain] /= total
        print("Done.")


    def load(self, filename):
        with open(os.path.expanduser(filename), "rb") as f:
            try:
                n, self.data = pickle.load(f)

                if self.order > n:
                    print("warning: changing order to", n)
                    self.order = n
                return True
            except:
                print("Loading data file failed!")
                return False

    def save(self, filename):
        try:
            with open(os.path.expanduser(filename), "wb") as f:
                pickle.dump((self.order, self.data), f)
            return True
        except:
            print("Could not save to file.")
            return False

    def reset(self, seed, prev):
        self.seed = seed
        self.prev = prev
        random.seed(seed)

    def __iter__(self):
        return self

    def next(self):
        next = self._nextToken(self.prev)
        if len(self.prev) > 0 and self.prev[0] == '':
            self.prev = []
        # accumulate a sentence, clearing it when it ends
        if next not in self.SENTENCE_ENDS:
            self.prev.append(next)
        else:
            self.prev = []
        return next

    def _nextToken(self, prev):
        prev = tuple(prev)

        if prev != ('',):
            while prev not in self.data:
                prev = prev[1:]
                if not prev:
                    return ''
        candidates = self.data[prev]
        sample = random.random()
        maxProb = 0.0
        bestToken = ""
        for candidate in candidates:
            prob = candidates[candidate]
            if prob > maxProb:
                maxProb = prob
                bestToken = candidate
            if sample > prob:
                sample -= prob
            else:
                return candidate
        # if we get here, we just make the best of it and take the best token found
        return bestToken

    # generate chunks where a chunk is defined as a series of tokens generated between
    # startPredicate and endPredicate returning true
    def generate(self, nchunks=1, seed=None, startPredicate=None, endPredicate=None, prefix=()):
        if not seed:
            seed = int(time.time())
        prefix = list(prefix) if prefix else []
        self.reset(seed, prefix)

        if not startPredicate:
            startPredicate = lambda t: True
        if not endPredicate:
            endPredicate = lambda t: True

        chunks = []
        while (nchunks > 0):
            # find a suitable token to start with
            token = next(self)
            while not startPredicate(token):
                token = next(self)

            pieces = [token]
            while True:
                token = next(self)
                # append punctuation to the previous token
                if not token.isalnum():
                    pieces[len(pieces)-1] += token
                else:
                    pieces.append(token)
                if not token or endPredicate(token):
                    nchunks -= 1
                    chunk = ' '.join(pieces)
                    # capitalize first letter
                    chunk = chunk.capitalize()
                    # fix end
                    if not chunk[len(chunk)-1] in Markov.SENTENCE_ENDS:
                        if chunk[len(chunk)-1] in Markov.CLAUSE_ENDS:
                            chunk = chunk[0:len(chunk) - 1]
                        chunk += random.choice(['!', '.'])
                    chunks.append(chunk)
                    # start over for a new chunk
                    self.prev = ['']
                    break
                    
        return ' '.join(chunks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # training args
    parser.add_argument('inputs', nargs='*')
    parser.add_argument('--paragraphs', action='store_true', default=False)
    parser.add_argument('--db', nargs='?', default="markov.db")
    parser.add_argument('--order', nargs='?', type=int, default=3)
    parser.add_argument('--reset', action='store_true', default=False)

    # generation args
    parser.add_argument('--sentence', action='store_true')
    parser.add_argument('--paragraph', action='store_true')
    parser.add_argument('--chunks', nargs='?', type=int, default=1)
    parser.add_argument('--tokens', nargs='?', type=int, default=0)
    parser.add_argument('--seed', nargs='?', type=int, default=int(time.time()))
    parser.add_argument('--prefix', nargs='?', default="")

    args = parser.parse_args()

    markov = Markov(args.order)

    corpus = []
    for path in args.inputs:
        corpus.extend(glob.glob(os.path.expanduser(path)))

    def chars(paths):
        for line in fileinput.input(paths, openhook=fileinput.hook_encoded("UTF-8")):
            for char in line:
                yield char

    if not args.reset and os.path.isfile(args.db):
        print("Loading db from " + args.db + "...")
        markov.load(args.db)
        print("Loaded db.")

    if len(corpus) > 0:
        tokens = Tokeniser(stream=chars(corpus), noparagraphs=not args.paragraphs)
        markov.train(tokens)
        print("Saving db to " + args.db + "...")
        markov.save(args.db)
        print("Saved db.")

    if args.paragraph:
        phrase = markov.generate(args.chunks, args.seed, 
                                 endPredicate=lambda t: t == '\n\n', 
                                 prefix=tuple(args.prefix.split(' ')))
        print(phrase)
    elif args.sentence:
        sentenceBoundary = lambda t: t[-1] in ".!?"
        phrase = markov.generate(args.chunks, args.seed,
                                 startPredicate=lambda t: t not in Markov.SENTENCE_ENDS,
                                 endPredicate=lambda t: t in Markov.SENTENCE_ENDS, 
                                 prefix=tuple(args.prefix.split(' ')))
        print(phrase)
    elif args.tokens:
        phrase = markov.generate(args.tokens, args.seed, prefix=tuple(args.prefix.split(' ')))
        print(phrase)
