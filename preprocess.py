
# -*- coding: utf8 -*-
import os
from parse import Parser
import numpy as np
import string
import nltk
import pickle
from nltk.tokenize import RegexpTokenizer

def get_review(fname):

    sent_tokenizer=nltk.data.load('tokenizers/punkt/english.pickle')
    tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
    with open("./data/stopWords.txt") as F:
        stopWords = set(map(str.strip, F.readlines()))
    s_tokens_list=[]
    w_tokens_list=[]
    with open(fname) as f:
        lines=f.readlines()
        for line in lines:

            line=line.strip().lower()
            sentences=sent_tokenizer.tokenize(line)
            s_tokens=[]
            w_tokens=[]
            for sentence in sentences:
                
                delEStr = string.punctuation + string.digits
                table = str.maketrans("", "",delEStr)
                words = tokenizer.tokenize(str(sentence))
                symbols = list(string.punctuation + string.digits)+list(stopWords)
                symbols.remove('!')
                tokens=[]
                for word in words:
                    if word not in symbols:
                        if word !='!':
                            word=word.translate(table)
                        if len(word) !=0:
                            tokens.append(word)
                if len(words) >0:
                    if len(words) == 1 and (words[0]=='!' or words[0] in stopWords):
                        pass
                    else:
                        s_tokens.append(tokens)
                        w_tokens.extend(tokens)
            s_tokens_list.append(s_tokens)
            w_tokens_list.append(w_tokens)
    return np.array(s_tokens_list),np.array(w_tokens_list)
def parseRawData(domain):
    work_dir=os.path.abspath(os.path.join(os.path.curdir,"data"))
    word_dir=os.path.abspath(os.path.join(work_dir,"word"))
    sentence_dir=os.path.abspath(os.path.join(work_dir,"sentence"))
    domain_dir1 = os.path.abspath(os.path.join(word_dir,domain))
    domain_dir2 = os.path.abspath(os.path.join(sentence_dir,domain))

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    if not os.path.exists(word_dir):
        os.makedirs(word_dir)
    if not os.path.exists(sentence_dir):
        os.makedirs(sentence_dir)
    if not os.path.exists(domain_dir1):
        os.makedirs(domain_dir1)
    if not os.path.exists(domain_dir2):
        os.makedirs(domain_dir2)

    fname ="./raw_data/%s/review_%s" %(domain,"positive")
    s_pos_data,w_pos_data= get_review(fname)

    fname ="./raw_data/%s/review_%s" %(domain,"negative")
    s_neg_data,w_neg_data= get_review(fname)

    pos_num, neg_num = len(s_pos_data), len(s_neg_data)
    np.random.seed(7)
    shuffle_pos_idx =np.random.permutation(np.arange(pos_num))
    s_pos_shuffle = s_pos_data[shuffle_pos_idx]
    s_pos_train = s_pos_shuffle[:2800]
    s_pos_test = s_pos_shuffle[2800:]
    write_s_tokensToFile(s_pos_train, domain, "train", "positive")
    write_s_tokensToFile(s_pos_test,  domain, "test",  "positive")

    w_pos_shuffle = w_pos_data[shuffle_pos_idx]
    w_pos_train = w_pos_shuffle[:2800]
    w_pos_test = w_pos_shuffle[2800:]
    write_w_tokensToFile(w_pos_train, domain, "train", "positive")
    write_w_tokensToFile(w_pos_test,  domain, "test",  "positive")

    shuffle_neg_idx = np.random.permutation(np.arange(neg_num))
    s_neg_shuffle = s_neg_data[shuffle_neg_idx]
    s_neg_train   = s_neg_shuffle[:2800]
    s_neg_test    = s_neg_shuffle[2800:]
    write_s_tokensToFile(s_neg_train, domain, "train", "negative")
    write_s_tokensToFile(s_neg_test,  domain, "test",  "negative")

    w_neg_shuffle   = w_neg_data[shuffle_neg_idx]
    w_neg_train     = w_neg_shuffle[:2800]
    w_neg_test      = w_neg_shuffle[2800:]
    write_w_tokensToFile(w_neg_train, domain, "train", "negative")
    write_w_tokensToFile(w_neg_test,  domain, "test",  "negative")

    fname ="./raw_data/%s/review_%s" %(domain,"unlabeled")
    s_data,w_data= get_review(fname)
    write_s_tokensToFile(s_data, domain, "train", "unlabeled")
    write_w_tokensToFile(w_data,  domain, "train",  "unlabeled")

def write_w_tokensToFile(tokens_list, domain, mode, label):

    fname = "./data/word/%s/tokens_%s.%s" % (domain, mode, label)
    print(fname, len(tokens_list))
    F = open(fname, 'w')
    for tokens in tokens_list:
        for token in tokens:
            F.write("%s " % token)
        F.write("\n")
    F.close()
    pass

def write_s_tokensToFile(h_tokens_list, domain, mode, label):

    fname = "./data/sentence/%s/tokens_%s.%s" % (domain, mode, label)
    print(fname, len(h_tokens_list))
    with open(fname, 'wb') as F:
        pickle.dump(h_tokens_list, F)
    F.close()
    pass

if __name__=="__main__":
    domains = ["books", "dvd", "electronics", "kitchen", "video"]
    for domain in domains:
        parseRawData(domain)