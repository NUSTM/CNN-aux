import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


class CNNnet(torch.nn.Module):
    def __init__(self):
        super(CNNnet,self).__init__()
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=100,
                            kernel_size=(3,300),
                            ),
            torch.nn.ReLU()
        )
        self.mlp3 = torch.nn.Linear(2*2*64,100)
        self.mlp4 = torch.nn.Linear(100,10)
    def forward(self, x):

        # x: (batch, sentence_length, embed_dim)
        x = x.unsqueeze(1)
        # x: (batch, 1, sentence_length,  )
        x = self.conv2(x)
        # x: (batch, kernel_num, H_out, 1)
        x = F.relu(x.squeeze(3))
        # x: (batch, kernel_num, H_out)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        #  (batch, kernel_num)
        return x

class CNN_SJ(nn.Module):
    def __init__(self, args,word2vecs):
        super(CNN_SJ, self).__init__()
        self.args=args
        self.CNN1 = CNNnet()
        self.CNN2 = CNNnet()
        self.word2vecs=word2vecs
        self.word_embedding=nn.Embedding(self.word2vecs.shape[0],self.word2vecs.shape[1])
        self.sentimentclassfier=torch.nn.Linear(2*args.hidden_size, 2)
        self.pivotposclassfier=torch.nn.Linear(args.hidden_size, 2)
        self.pivotnegclassfier=torch.nn.Linear(args.hidden_size, 2)
        # self.dropout = nn.Dropout(0.5)
        self.reset_parameters()

    def forward(self,main_inputs,au_inputs,u_labels,v_labels,flag,main_labels=None):
        loss_fct = CrossEntropyLoss()
        if flag==2:
            au_em = self.word_embedding(au_inputs)
            au_hidden = self.CNN2(au_em)
            main_em = self.word_embedding(main_inputs)
            main_hidden = self.CNN1(main_em)
            joint = torch.cat([au_hidden, main_hidden], 1)    
            s_logit= self.sentimentclassfier(joint)
            return s_logit
        elif flag==0:
            au_em = self.word_embedding(au_inputs)
            main_em = self.word_embedding(main_inputs)
            main_hidden = self.CNN1(main_em)
            au_hidden = self.CNN2(au_em)
            # au_hidden = self.dropout(au_hidden)
            pos = self.pivotposclassfier(au_hidden)
            neg = self.pivotnegclassfier(au_hidden)
            au_loss = loss_fct(pos, u_labels) + loss_fct(neg, v_labels)
            joint=torch.cat([au_hidden, main_hidden], 1)
            # joint = self.dropout(joint)
            main_loss=loss_fct(self.sentimentclassfier(joint),main_labels)
            return main_loss,au_loss

        elif flag==1:
            au_em = self.word_embedding(au_inputs)
            au_hidden = self.CNN2(au_em)
            # au_hidden = self.dropout(au_hidden)
            pos = self.pivotposclassfier(au_hidden)
            neg = self.pivotnegclassfier(au_hidden)
            au_loss = loss_fct(pos, u_labels) + loss_fct(neg, v_labels)
            return au_loss
    def reset_parameters(self):
        self.word2vecs=torch.tensor(self.word2vecs).float()
        self.word_embedding.weight.data=self.word2vecs


