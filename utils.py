import numpy as np
from numpy import *
import os
import re
import pickle
import nltk
import torch
import math
def vectorize_data(data, pos_pivot, neg_pivot, word2idx, memory_size, sentence_size):

    S = []
    Q = []
    domains = []
    word_mask = []
    sentence_mask = []
    u_labels  = []
    v_labels  = []
    for review,domain,label in data:
        ss=[]
        mask=[]
        u_label  = 0
        v_label  = 0
        for sentence in review:
            mask_id=[]
            for word in sentence:
                if word in pos_pivot:
                    mask_id.append(0)
                    u_label  = 1
                elif word in neg_pivot:
                    mask_id.append(0)
                    v_label  = 1
                else:
                    mask_id.append(1)
            ls=max(0,sentence_size-len(sentence))
            if ls==0:
                ss.append([word2idx[word] for word in sentence][:sentence_size])
                mask_id=mask_id[:sentence_size]
            else:
                ss.append([word2idx[word] for word in sentence]+[0]*ls)
                mask_id=mask_id+[0]*ls
            mask.append(mask_id)
        ss=ss[:memory_size]
        mask=mask[:memory_size]
        sent=[]
        lm=max(0,memory_size-len(ss))
        sent=[1]*len(ss)+[0]*lm
        ss.extend([[0]*sentence_size]*lm)
        mask.extend([[0]*sentence_size]*lm)
        
    
        S.append(ss)
        Q.append(label)
        domains.append(domain)
        word_mask.append(mask)
        sentence_mask.append(sent)
        u_labels.append(u_label)
        v_labels.append(v_label)
    return torch.tensor(S), torch.tensor(domains), torch.tensor(Q),torch.tensor(u_labels),torch.tensor(v_labels), torch.tensor(word_mask), torch.tensor(sentence_mask)

def cnn_vectorize_data(data, pos_pivot, neg_pivot, word2idx, sentence_size):

    S = []
    Q = []
    domains = []
    word_mask = []

    u_labels  = []
    v_labels  = []
    pos_pivot={}.fromkeys(pos_pivot)
    neg_pivot={}.fromkeys(neg_pivot)
    for review,domain,label in data:
        ss=[]
        mask=[]
        u_label  = 0
        v_label  = 0
        for sentence in review:
            mask_id=[]
            for word in sentence:
                if word in pos_pivot:
                    mask_id.append(0)
                    u_label  = 1
                elif word in neg_pivot:
                    mask_id.append(0)
                    v_label  = 1
                else:
                    mask_id.append(1)
            
            ss.extend([word2idx[word] for word in sentence])
            mask.extend(mask_id)
        ss=ss[:sentence_size]
        mask=mask[:sentence_size]
 
        lm=max(0,sentence_size-len(ss))
     
        ss.extend([0]*lm)
        mask.extend([0]*lm)
        S.append(ss)
        Q.append(label)
        domains.append(domain)
        word_mask.append(mask)
        u_labels.append(u_label)
        v_labels.append(v_label)
    return torch.tensor(S), torch.tensor(domains), torch.tensor(Q),torch.tensor(u_labels),torch.tensor(v_labels), torch.tensor(word_mask)

def filter_byMI(samp_list_train,term_dict,pivot_term,domain):
    pos=0
    neg=0
    

    term_class={}
    # for term in term_dict.keys():
    #     term_class[term]=[0]*2

    for k in range(len(samp_list_train)):
        review, _, label=samp_list_train[k]
        
        if label==0:
            neg+=1
            for sentence in review:
                pos_sentence = nltk.pos_tag(sentence)
                for i,term in enumerate(sentence):
                    term=term+'_'+pos_sentence[i][1]
                    if term not in term_class:
                        term_class[term]=[0]*2
                        term_class[term][0]+=1
                    else:
                        term_class[term][0]+=1
        else:
            pos+=1
            for sentence in review:
                pos_sentence = nltk.pos_tag(sentence)
                for i,term in enumerate(sentence):
                    term=term+'_'+pos_sentence[i][1]
                    if term not in term_class:
                        term_class[term]=[0]*2
                        term_class[term][1]+=1
                    else:
                        term_class[term][1]+=1
            
    # term_set = term_dict.keys()
    pos_term_score_dict = {}
    neg_term_score_dict = {}
    N = pos+neg
    for term,count in term_class.items():
        pos_a=term_class[term][1]
        neg_a=term_class[term][0]
       
        pos_c_t = (pos_a+1.0)/(pos_a+neg_a+2.0)
        pos_c= float(pos)/N
        neg_c_t=(neg_a+1.0)/(pos_a+neg_a+2.0)
        neg_c= float(neg)/N
        pos_score = math.log(pos_c_t/pos_c)
        neg_score = math.log(neg_c_t/neg_c) 
        pos_term_score_dict[term] = pos_score
        neg_term_score_dict[term] = neg_score
    pos_term_score_dict=sorted(pos_term_score_dict.items(),key = lambda x:-x[1])    
    neg_term_score_dict=sorted(neg_term_score_dict.items(),key = lambda x:-x[1])  

    pos_term = [x[0] for x in pos_term_score_dict]   
    neg_term = [x[0] for x in neg_term_score_dict]
    pos_term,neg_term=unis_filter(neg_term,pos_term,0.125,pivot_term)
    cnn_store_pivots(pos_term,neg_term, domain)
    return pos_term,neg_term
def filter_byWLLR(samp_list_train,term_dict,pivot_term,domain):
    pos=0
    neg=0
    

    term_class={}
    # for term in term_dict.keys():
    #     term_class[term]=[0]*2

    for k in range(len(samp_list_train)):
        review, _, label=samp_list_train[k]
        
        if label==0:
            neg+=1
            for sentence in review:
                pos_sentence = nltk.pos_tag(sentence)
                for i,term in enumerate(sentence):
                    term=term+'_'+pos_sentence[i][1]
                    if term not in term_class:
                        term_class[term]=[0]*2
                        term_class[term][0]+=1
                    else:
                        term_class[term][0]+=1
        else:
            pos+=1
            for sentence in review:
                pos_sentence = nltk.pos_tag(sentence)
                for i,term in enumerate(sentence):
                    term=term+'_'+pos_sentence[i][1]
                    if term not in term_class:
                        term_class[term]=[0]*2
                        term_class[term][1]+=1
                    else:
                        term_class[term][1]+=1
            
    # term_set = term_dict.keys()
    pos_term_score_dict = {}
    neg_term_score_dict = {}
    N = pos+neg
    for term,count in term_class.items():
        pos_a=term_class[term][1]
        neg_a=term_class[term][0]
       
        c_y = (pos_a+1E-6)/(float(pos)+1E-6*2)
        c_n_y=(neg_a+1E-6)/(float(neg)+1E-6*2)

        pos_score = c_y*math.log(c_y/c_n_y)
        neg_score = c_n_y*math.log(c_n_y/c_y) 
        pos_term_score_dict[term] = pos_score
        neg_term_score_dict[term] = neg_score
    pos_term_score_dict=sorted(pos_term_score_dict.items(),key = lambda x:-x[1])    
    neg_term_score_dict=sorted(neg_term_score_dict.items(),key = lambda x:-x[1])  

    pos_term = [x[0] for x in pos_term_score_dict]   
    neg_term = [x[0] for x in neg_term_score_dict]
    pos_term,neg_term=unis_filter(neg_term,pos_term,0.25,pivot_term)
    cnn_store_pivots(pos_term,neg_term, domain)
    return pos_term,neg_term
def cnn_store_pivots(pos_pivots, neg_pivots, domain):

    output_dir = "./work/cnn_pivots/"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    fname = "./work/cnn_pivots/" + domain +"_pos.txt"
    print(fname)

    with open(fname, "w") as f:
        for key in pos_pivots:
            f.write("%s\n" % key)
    f.close()

    fname = "./work/cnn_pivots/" + domain +"_neg.txt"
    print(fname)

    with open(fname, "w") as f:
        for key in neg_pivots:
            f.write("%s\n" % key)
    f.close()
def unis_filter(neg, pos, fs_percent,pivot_term):
    postag_list= ['JJ','JJS','JJR','RB','RBS','RBR','VB','VBZ','VBD','VBN','VBG','VBP']
    adverse_list = ['not', 'no', 'without', 'never', 'n\'t', 'don\'t', 'hardly']
    neg_postag_list = pos_list_filter(neg, adverse_list, postag_list,pivot_term)
    pos_postag_list = pos_list_filter(pos, adverse_list, postag_list,pivot_term)
    dict_len = int(min(len(neg_postag_list), len(pos_postag_list)) * fs_percent)
    return pos_postag_list[:dict_len],neg_postag_list[:dict_len]


def pos_list_filter(pol_list, adverse_list, pos_list,pivot_term):
    filter_pos_list = []
    pivot_term = {}.fromkeys(pivot_term,0)
    import nltk
    from nltk.corpus import stopwords
    stopwords = stopwords.words('english')
    stopword_short = ["'m", "'s", "'re", "'ve", "e", "d"]
    for k in range(len(pol_list)):
        term = pol_list[k]
        if '_' not in term:
            continue
        term_list = term.split('_')
        if term_list[0] in pivot_term and term_list[1] in pos_list and term_list[0] not in adverse_list and term_list[0] not in stopwords and \
                term_list[0] not in stopword_short:
            if pivot_term[term_list[0]]==0:
                filter_pos_list.append(term_list[0])
                pivot_term[term_list[0]]+=1
    return filter_pos_list
def add_unknown_words(word_vecs, vocab, min_df=1, dim=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, dim)

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                # print(ch)
                if ch == b' ':
                    word = ''.join(word)

                    break
                if ch != '\n':
                    word.append(ch.decode('cp437'))
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs



def get_w2vec(vocab, args):
    word_vecs=load_bin_vec(args.w2v_path,vocab)
    add_unknown_words(word_vecs, vocab)
    dim=np.array(list(word_vecs.values())).shape[1]
    vocab_size=len(word_vecs)   
    word_idx_map=dict(list(zip(word_vecs.keys(), list(range(1, 1 + len(word_vecs))))))
    idx_word_map=dict(list(zip(list(range(1, 1 + len(word_vecs))),word_vecs.keys())))
    
    W=np.zeros(shape=(vocab_size+1,dim),dtype='float32')
    W[0] =np.zeros(dim,dtype='float32')
    W[1:,:]=list(word_vecs.values())
    return W,word_idx_map,idx_word_map
   
def getVocab(all_data):
    vocab={}
    for review, _, _, in all_data:
        for sentence in review:
            for word in sentence:
                vocab[word]=vocab.get(word,0)+1
    return vocab
    
def get_review(f, domain, label):
    reviews=[]
    y=1
    if label=="positive":
        y=1
    elif label=="negative":
        y=0
    with open(f,'rb') as F:
        token_list=pickle.load(F)
        for tokens in token_list:
            reviews.append((tokens,domain,y))
    return reviews

def load_data(source_domain,target_domain,root_path):
    train_data = []
    test_data = []
    val_data = []
    source_unlabeled_data = []
    target_unlabeled_data = []
    src, tar = 1, 0

    print ("source domain: ", source_domain, "target domain:", target_domain)

    #load training data
    for (mode,label) in [("train","positive"),("train","negative")]:
        fname =root_path+"%s/tokens_%s.%s" % (source_domain,mode,label)
        train_data.extend(get_review(fname,src,label))
    print("train_size:",len(train_data))

    #load validation data
    for (mode,label) in [("test","positive"),("test","negative")]:
        fname =root_path+"%s/tokens_%s.%s" % (source_domain,mode,label)
        val_data.extend(get_review(fname,src,label))
    print("val_size:",len(val_data))

    #load test data
    for (mode,label) in [("train","positive"),("train","negative"),("test","positive"),("test","negative")]:
        fname =root_path+"%s/tokens_%s.%s" % (target_domain,mode,label)
        test_data.extend(get_review(fname,tar,label))
    print("test_size:",len(test_data))

    #load unlabeled data
    for (mode,label) in [("train","unlabeled")]:
        fname =root_path+"%s/tokens_%s.%s" % (source_domain,mode,label)
        source_unlabeled_data.extend(get_review(fname,src,label))
    print("source_unlabeled_size:",len(source_unlabeled_data))
    for (mode,label) in [("train","unlabeled")]:
        fname =root_path+"%s/tokens_%s.%s" % (target_domain,mode,label)
        target_unlabeled_data.extend(get_review(fname,tar,label))
    print("target_unlabeled_size:",len(target_unlabeled_data))
    
    #build vocab
    vocab = getVocab(train_data + val_data + test_data + source_unlabeled_data + target_unlabeled_data)
    print("vocab-size: ", len(vocab))

    output_dir = "./work/logs/"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    return train_data, val_data, test_data, source_unlabeled_data, target_unlabeled_data, vocab

def cnn_load_data(source_domain,target_domain,root_path):
    train_data = []
    test_data = []
    val_data = []
    source_unlabeled_data = []
    target_unlabeled_data = []
    src, tar = 1, 0

    print ("source domain: ", source_domain, "target domain:", target_domain)

    #load training data
    for (mode,label) in [("train","positive"),("train","negative")]:
        fname =root_path+"%s/tokens_%s.%s" % (source_domain,mode,label)
        train_data.extend(get_review(fname,src,label))
    print("train_size:",len(train_data))

    #load validation data
    for (mode,label) in [("test","positive"),("test","negative")]:
        fname =root_path+"%s/tokens_%s.%s" % (source_domain,mode,label)
        val_data.extend(get_review(fname,src,label))
    print("val_size:",len(val_data))

    #load test data
    for (mode,label) in [("train","positive"),("train","negative"),("test","positive"),("test","negative")]:
        fname =root_path+"%s/tokens_%s.%s" % (target_domain,mode,label)
        test_data.extend(get_review(fname,tar,label))
    print("test_size:",len(test_data))

    #load unlabeled data
    for (mode,label) in [("train","unlabeled")]:
        fname =root_path+"%s/tokens_%s.%s" % (source_domain,mode,label)
        source_unlabeled_data.extend(get_review(fname,src,label))
    print("source_unlabeled_size:",len(source_unlabeled_data))
    for (mode,label) in [("train","unlabeled")]:
        fname =root_path+"%s/tokens_%s.%s" % (target_domain,mode,label)
        target_unlabeled_data.extend(get_review(fname,tar,label))
    print("target_unlabeled_size:",len(target_unlabeled_data))
    
    #build vocab
    source_vocab=getVocab(train_data+source_unlabeled_data+ val_data)
    target_vocab=getVocab(target_unlabeled_data+ test_data)
    vocab = {}
    for key in list(set(source_vocab) | set(target_vocab)):
        if source_vocab.get(key) and target_vocab.get(key):
            vocab.update({key: source_vocab.get(key) + target_vocab.get(key)})
        else:
            vocab.update({key: source_vocab.get(key) or target_vocab.get(key)})
    print("vocab-size: ", len(vocab))

    output_dir = "./work/logs/"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    return train_data, val_data, test_data, source_unlabeled_data, target_unlabeled_data, source_vocab,target_vocab,vocab
import torch.nn.functional as F
def mask_softmax(target, axis, mask, epsilon=1e-12, name=None):   
    
    max_axis = torch.tensor(target.max(axis,keepdim=True)[0].data)
    target_exp = torch.mul(torch.exp(torch.sub(target,max_axis)),mask)
    normalize = target_exp.sum(axis,keepdim=True)
    softmax = target_exp / (normalize + epsilon)
    return softmax

def get_pivot(source_domain,target_domain):

        nfname = "./work/pivots/" + source_domain+"_"+target_domain +"_neg.txt"

        pfname = "./work/pivots/" + source_domain+"_"+target_domain +"_pos.txt"
        n_word=[]
        with open(nfname,"r")as f:
            lines=f.readlines()
            for line in lines:
                n_word.append(line.strip().split(" ")[0])
        p_word=[]
        with open(pfname,"r")as f:
            lines=f.readlines()
            for line in lines:
                p_word.append(line.strip().split(" ")[0])
        return n_word,p_word

def feature_selection_df(df_term, thrd):
    term_df=[]
    for term in df_term.keys():
        if df_term[term]>=thrd:
            term_df.append(term)
    return term_df


def c_l2_regularization(model,lamb):
    l2_regularization = torch.tensor([0],dtype=torch.float32).cuda()
    l2_regularization += torch.norm(model.pivotposclassfier.weight.data, 2) 
    l2_regularization += torch.norm(model.pivotnegclassfier.weight.data, 2) 
    l2_regularization += torch.norm(model.sentimentclassfier.weight.data, 2) 
    return lamb*l2_regularization

def c_l2_regularization_v(model,lamb):
    l2_regularization = torch.tensor([0],dtype=torch.float32).cuda()
    l2_regularization += torch.norm(model.sentimentclassfier.weight.data, 2) 
    return lamb*l2_regularization