#!/usr/bin/env python

import random
import pickle
import os
import sys
import argparse
import time
import glob
import fileinput

class Tokeniser:
    """Flexible tokeniser for the Markov chain.
    """

    def __init__(self, stream=None, noparagraphs=False):
        self.stream = sys.stdin if stream is None else stream
        self.noparagraphs = noparagraphs

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
                cout = False

                if self.buffer == '\n' and next_char == '\n':
                    # Paragraph break
                    if not self.noparagraphs:
                        out = self.buffer + next_char
                        next_char = ''

                elif not self.buffer.isspace() and next_char.isspace():
                    # A word
                    out = self.buffer

                # If the next_char is a token, save it
                if cout:
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
class Markov:
    CLAUSE_ENDS = [',', '.', ';', ':']

    # n = prefix length
    def __init__(self, n=3):
        self.n = n
        self.p = 0
        self.seed = None
        self.data = {}
        self.cln = n

    def set_cln(self, cln):
        self.cln = cln if cln is not None and cln <= self.n else self.n

    def train(self, tokenStream):
        prev = ()
        for token in tokenStream:
            #token = sys.intern(token)
            for pprev in [prev[i:] for i in range(len(prev) + 1)]:
                if not pprev in self.data:
                    self.data[pprev] = [0, {}]

                if not token in self.data[pprev][1]:
                    self.data[pprev][1][token] = 0

                self.data[pprev][1][token] += 1
                self.data[pprev][0] += 1

            prev += (token,)
            if len(prev) > self.n:
                prev = prev[1:]

    def load(self, filename):
        with open(os.path.expanduser(filename), "rb") as f:
            try:
                n, self.data = pickle.load(f)

                if self.n > n:
                    print("warning: changing n value to", n)
                    self.n = n
                return True
            except:
                print("Loading data file failed!")
                return False

    def save(self, filename):
        try:
            with open(os.path.expanduser(filename), "wb") as f:
                pickle.dump((self.n, self.data), f)
            return True
        except:
            print("Could not save to file.")
            return False

    def reset(self, seed, prob, prev, cln):
        self.seed = seed
        self.p = prob
        self.prev = prev
        self.set_cln(cln)
        random.seed(seed)

    def __iter__(self):
        return self

    def next(self):
        if self.prev == () or random.random() < self.p:
            next = self._choose(self.data[()])
        else:
            try:
                next = self._choose(self.data[self.prev])
            except:
                self.prev = ()
                next = self._choose(self.data[self.prev])

        self.prev += (next,)
        if len(self.prev) > self.n:
            self.prev = self.prev[1:]

        if next[-1] in self.CLAUSE_ENDS:
            self.prev = self.prev[-self.cln:]

        return next

    def _choose(self, freqdict):
        total, choices = freqdict
        idx = random.randrange(total)

        for token, freq in choices.items():
            if idx <= freq:
                return token

            idx -= freq

    def generate(self, chunks=1, seed=None, randTokenChance=0, startPredicate=None, endPredicate=None, prefix=()):
    	if not seed:
    		seed = int(time.time())
    	self.reset(seed, randTokenChance, prefix, None)

    	if not startPredicate:
    		startPredicate = lambda t: True
    	if not endPredicate:
    		endPredicate = lambda t: True

    	while not startPredicate(next(self)):
    		pass

    	def gen(n):
    		out = []
    		while (n > 0):
    			token = next(self)
    			out.append(token)
    			if endPredicate(token):
    				n -= 1
    		return ' '.join(out)

    	self.generator = gen
    	return self.generator(chunks)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	# training args
	parser.add_argument('inputs', nargs='*')
	parser.add_argument('--paragraphs', action='store_true', default=False)
	parser.add_argument('--db', nargs='?', default="markov.db")
	parser.add_argument('--length', nargs='?', type=int, default=3)

	# generation args
	parser.add_argument('--sentence', action='store_true')
	parser.add_argument('--paragraph', action='store_true')
	parser.add_argument('--chunks', nargs='?', type=int, default=1)
	parser.add_argument('--tokens', nargs='?', type=int, default=0)
	parser.add_argument('--rand', nargs='?', type=int, default=1)
	parser.add_argument('--seed', nargs='?', type=int, default=int(time.time()))
	parser.add_argument('--prefix', nargs='?', default="")

	args = parser.parse_args()

	markov = Markov(args.length)

	corpus = []
	for path in args.inputs:
		corpus.extend(glob.glob(os.path.expanduser(path)))

	def chars(paths):
		for line in fileinput.input(paths):
			for char in line:
				yield char

	if len(corpus) > 0:
		tokens = Tokeniser(stream=chars(corpus), noparagraphs=not args.paragraphs)
		markov.train(tokens)
		print("Saving db to " + args.db + "...")
		markov.save(args.db)
		print("Saved db.")
	else:
		print("Loading db from " + args.db + "...")
		markov.load(args.db)
		print("Loaded db.")

	if args.paragraph:
		phrase = markov.generate(args.chunks, args.seed, args.rand, endPredicate=lambda t: t == '\n\n', prefix=tuple(args.prefix.split(' ')))
		print(phrase)
	elif args.sentence:
		sentenceBoundary = lambda t: t[-1] in ".!?"
		phrase = markov.generate(args.chunks, args.seed, args.rand, sentenceBoundary, sentenceBoundary, prefix=tuple(args.prefix.split(' ')))
		print(phrase)
	elif args.tokens:
		phrase = markov.generate(args.tokens, args.seed, args.rand, prefix=tuple(args.prefix.split(' ')))
		print(phrase)
