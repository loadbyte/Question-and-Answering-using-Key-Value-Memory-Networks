'''Description:- Here we are creating utility functions that are being used frequently '''

from __future__ import absolute_import

import os
import re
import numpy as np


''' Below method vectorizes the data set partitionning them into story,question and answer tuple. For stories we are considering a vector of size of maximum
 length story * the size of length of maximum sentence or a threshold value provided at the time of training the stories which is not having enough sentences 
 are padded with zeros. Similarly for sentences not having enough length are padded with 0. Answer vector is created as a 1-D vector of vocabulary size+1 and 
 have a 1 in place of the index of answer word other cells are field with 0. '''
def vectorize_data(data, word_idx, sentence_size, memory_size):
    S = []
    Q = []
    A = []
    for story, query, answer in data:
        ss = []
        for i, sentence in enumerate(story, 1):
            ls = max(0, sentence_size - len(sentence))
            ss.append([word_idx[w] for w in sentence] + [0] * ls)

        # take only the most recent sentences that fit in memory
        ss = ss[::-1][:memory_size]

        # pad to memory_size
        lm = max(0, memory_size - len(ss))
        for _ in range(lm):
            ss.append([0] * sentence_size)

        lq = max(0, sentence_size - len(query))
        q = [word_idx[w] for w in query] + [0] * lq

        y = np.zeros(len(word_idx) + 1) # 0 is reserved for nil word
        for a in answer:
            y[word_idx[a]] = 1

        S.append(ss)
        Q.append(q)
        A.append(y)
    return np.array(S), np.array(Q), np.array(A)

# Below method tokenizes sentence in words including stop words.
def word_tokenize(sent):
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

# Below method get the lines of dataset file as input and extracts substories , its related question and answer.
def parse_stories(lines):
    data = []
    story = []
    for line in lines:
        line = str.lower(line)
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line: # question
            q, a, supporting = line.split('\t')
            q = word_tokenize(q)
            #a = tokenize(a)
            # answer is one vocab word even if it's actually multiple words
            a = [a]
            substory = None

            # remove question marks
            if q[-1] == "?":
                q = q[:-1]

            substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else: # regular sentence
            # remove periods
            sent = word_tokenize(line)
            if sent[-1] == ".":
                sent = sent[:-1]
            story.append(sent)
    return data

#Below method reads from the dataset text file and send it for parsing
def get_stories(f):
    with open(f) as f:
        return parse_stories(f.readlines())

''' Below method works as a pilot method which checks if we are getting valid task file and then it calls for sub functions to split the dataset as story, question 
and answer tuples'''
def load_task(data_dir, task_id):
    
    assert task_id > 0 and task_id < 21
    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    s = 'qa{}_'.format(task_id)
    train_file = [f for f in files if s in f and 'train' in f][0]
    test_file = [f for f in files if s in f and 'test' in f][0]
    train_data = get_stories(train_file)
    test_data = get_stories(test_file)
    return train_data, test_data


