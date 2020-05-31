# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# #
# # Copyright (C) 2016 Manas Ranjan Kar <manasrkar91@gmail.com>
# # Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
#
# """
# CLI USAGE: python glove2word2vec.py <GloVe vector file> <Output model file>
# Convert GloVe vectors into word2vec C format;
# model = gensim.models.Word2Vec.load_word2vec_format('/tmp/vectors.txt', binary=False)  # C text format
# word2vec embeddings start with a line with the number of lines (tokens?) and the number of dimensions of the file. This allows gensim to allocate memory
# accordingly for querying the model. Larger dimensions mean larger memory is held captive. Accordingly, this line has to be inserted into the GloVe
# embeddings file.
# """
#
#
# import re
# import sys
# import gensim
# import logging
# import smart_open
#
# program=sys.argv[0]
# logger = logging.getLogger(program)
# logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
#
# def glove2word2vec(glove_vector_file, output_model_file):
#     """Convert GloVe vectors into word2vec C format"""
#
#     def get_info(glove_file_name):
#         """Return the number of vectors and dimensions in a file in GloVe format."""
#         with smart_open.smart_open(glove_file_name) as f:
#             num_lines = sum(1 for line in f)
#         with smart_open.smart_open(glove_file_name) as f:
#             num_dims = len(f.readline().split()) - 1
#         return num_lines, num_dims
#
#     def prepend_line(infile, outfile, line):
#         """
#         Function to prepend lines using smart_open
#         """
#         with smart_open.smart_open(infile, 'rb') as old:
#             with smart_open.smart_open(outfile, 'wb') as new:
#                 new.write(str(line.strip()) + "\n")
#                 for line in old:
#                     new.write(line)
#         return outfile
#
#     num_lines, dims = get_info(glove_vector_file)
#
#     logger.info('%d lines with %s dimensions' % (num_lines, dims))
#
#     gensim_first_line = "{} {}".format(num_lines, dims)
#     model_file = prepend_line(glove_vector_file, output_model_file, gensim_first_line)
#
#     logger.info('Model %s successfully created !!'%output_model_file)
#
#     # Demo: Loads the newly created glove_model.txt into gensim API.
#     model = gensim.models.Word2Vec.load_word2vec_format(model_file, binary=False) #GloVe Model
#
#     logger.info('Most similar to king are: %s' % model.most_similar(positive=['king'], topn=10))
#     logger.info('Similarity score between woman and man is %s ' % model.similarity('woman', 'man'))
#
#     logger.info("Finished running %s", program)
#
#     return model_file
#
# if __name__ == "__main__":
#
#     input_file = 'data/glove.twitter.27B.50d.txt'
#     output_file = 'data/glove.twitter.27B.50d_w2v.txt'
#     glove2word2vec(input_file, output_file)


# -*- coding: utf-8 -*-
"""
word2vec embeddings start with a line with the number of lines (tokens?) and
the number of dimensions of the file. This allows gensim to allocate memory
accordingly for querying the model. Larger dimensions mean larger memory is
held captive. Accordingly, this line has to be inserted into the GloVe
embeddings file.
"""

import os
import shutil
import smart_open
from sys import platform

import gensim


def prepend_line(infile, outfile, line):
    """
    Function use to prepend lines using bash utilities in Linux.
    (source: http://stackoverflow.com/a/10850588/610569)
    """
    with open(infile, 'r') as old:
        with open(outfile, 'w') as new:
            new.write(str(line) + "\n")
            shutil.copyfileobj(old, new)


def prepend_slow(infile, outfile, line):
    """
    Slower way to prepend the line by re-creating the inputfile.
    """
    with open(infile, 'r') as fin:
        with open(outfile, 'w') as fout:
            fout.write(line + "\n")
            for line in fin:
                fout.write(line)


def get_lines(glove_file_name):
    """Return the number of vectors and dimensions in a file in GloVe format."""
    with smart_open.smart_open(glove_file_name, 'r') as f:
        num_lines = sum(1 for line in f)
    with smart_open.smart_open(glove_file_name, 'r') as f:
        num_dims = len(f.readline().split()) - 1
    return num_lines, num_dims


# Input: GloVe Model File
# More models can be downloaded from http://nlp.stanford.edu/projects/glove/
glove_file = 'data/glove.twitter.27B.50d.txt'

num_lines, dims = get_lines(glove_file)

# Output: Gensim Model text format.
gensim_file = 'data/glove.twitter.27B.50d_w2v.txt'
gensim_first_line = "{} {}".format(num_lines, dims)

# Prepends the line.
if platform == "linux" or platform == "linux2":
    prepend_line(glove_file, gensim_file, gensim_first_line)
else:
    prepend_slow(glove_file, gensim_file, gensim_first_line)

# Demo: Loads the newly created glove_model.txt into gensim API.
model = gensim.models.KeyedVectors.load_word2vec_format(gensim_file, binary=False)  # GloVe Model

print(model.most_similar(positive=['australia'], topn=10))
print(model.similarity('woman', 'man'))