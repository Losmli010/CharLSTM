# coding: utf-8

import sys
from datetime import datetime
import numpy as np
import re
from utils import *

def tokenizer(sentence):
    re_chinese=re.compile(r"([\u4E00-\u9FA5]+)")
    re_english=re.compile(r"([0-9A-Za-z]+)")
    block=re.split(re_chinese,sentence)
    tokens=[]
    for s in block:
        if re.match(re_chinese,s):
            tokens +=[s[i] for i in range(len(s))]
        else:
            tem=re.split(re_english,s)
            tokens +=[x for x in tem if x]
    return tokens

def getSentenceData(path):
    sentence_start_token = "START"
    sentence_end_token = "END"
    
    f=open(path,"r",encoding="utf-8")
    text=[line.strip() for line in f]
    f.close()
    sentences = ["%s%s%s" % (sentence_start_token, x, sentence_end_token) for x in text]
    print("Parsed %d sentences." % (len(sentences)))
    
    sent_tokens=[tokenizer(sent) for sent in sentences]
    chars_set=sorted(list(set([char for sent in sent_tokens for char in sent])))
    print("Found %d unique chars tokens." % len(chars_set))
    
    char2index=dict(zip(chars_set,np.arange(len(chars_set))))
    index2char=dict(zip(np.arange(len(chars_set)),chars_set))
    
    X_train = np.asarray([[char2index[c] for c in sent[:-1]] for sent in sent_tokens])
    y_train = np.asarray([[char2index[c] for c in sent[1:]] for sent in sent_tokens])
    print("X_train shape: " + str(X_train.shape))
    print("y_train shape: " + str(y_train.shape))

    # Print an training data example
    x_example, y_example = X_train[2], y_train[2]
    print("x:\n%s\n%s" % (" ".join([index2char[x] for x in x_example]), x_example))
    print("\ny:\n%s\n%s" % (" ".join([index2char[x] for x in y_example]), y_example))
    return X_train, y_train ,index2char, char2index

def train(model, X, Y, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
    num_examples_seen = 0
    losses = []
    for epoch in range(nepoch):
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_total_loss(X, Y)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss))
            # Adjust the learning rate if loss increases
            if len(losses) > 1 and losses[-1][1] > losses[-2][1]:
                learning_rate = learning_rate * 0.5
                print("Setting learning rate to %f" % learning_rate)
            sys.stdout.flush()
        # For each training example...
        for i in range(len(Y)):
            model.sgd_step(X[i], Y[i], learning_rate)
            num_examples_seen += 1
        if epoch%50==0:
            save_model_parameters(model, 'data/lstmlm.parameters.epoch%s.npz'%epoch)

def print_sentence(sent, index2char):
    sentence_str = [index2char[x] for x in sent[1:-1]]
    print("".join(sentence_str))
    sys.stdout.flush()

def generate_sentence(model, index2char, char2index):
    new_sentence = [char2index["START"]]
    while not new_sentence[-1] == char2index["END"]:
        next_word= model.predict(new_sentence)[-1]
        new_sentence.append(next_word)
        if len(new_sentence) > 100:
            return None
    return new_sentence

def generate_sentences(model, n, index2char, char2index):
    for i in range(n):
        sent=None
        while not sent:
            sent = generate_sentence(model, index2char, char2index)
        print_sentence(sent, index2char)   
