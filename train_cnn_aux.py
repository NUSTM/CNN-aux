import matplotlib
matplotlib.use('Agg')
from sklearn import metrics
import argparse
import sys

import math
import torch
from utils import *
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
from model.cnn_aux import *
from torch.nn import init
from model.log import Logger
sys.stdout = Logger("./work/logs/cnn_aux.txt") #这里我将Log输出到D盘
# from models import PNet
def adjust_learning_rate(optimizer, p):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = max(0.005/math.pow(1+10*p,0.75),0.002)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--source_domain', '-s', type=str,
                        choices=['books', 'dvd', 'electronics', 'kitchen', 'video'],
                        default='electronics')
    parser.add_argument('--target_domain', '-t', type=str,
                        choices=['books', 'dvd', 'electronics', 'kitchen', 'video'],
                        default='kitchen')
    parser.add_argument('--data_path',default='./data/sentence/',type=str,required=False)
    parser.add_argument('--memory_size',default=20,type=int,required=False)
    parser.add_argument('--sent_size',default=25,type=int,required=False)
    parser.add_argument('--embed_size',default=300,type=int,required=False)
    parser.add_argument('--hidden_size',default=100,type=int,required=False)

    parser.add_argument('--w2v_path',default='../1.bin',type=str,required=False)
    parser.add_argument('--random_seed',default=0,type=int,required=False)
    parser.add_argument('--train_batch_size',default=50,type=int,required=False)
    parser.add_argument('--domain_batch_size',default=50,type=int,required=False)
    parser.add_argument('--num_train_epochs',default=100,type=int,required=False)
    parser.add_argument('--val_interval',default=1,type=int,required=False)
  
    parser.add_argument('--lr',default=0.005,type=double,required=False)
    parser.add_argument('--l2_reg_lambda',default=0.2,type=double,required=False)
    parser.add_argument('--penalty',default=0.005,type=double,required=False)
    parser.add_argument('--patience',default=5,type=int,required=False)

    args = parser.parse_args()
    np.random.seed(args.random_seed)
    source_domain = args.source_domain
    target_domain = args.target_domain
    data_path     = args.data_path

    print("loading data...")
    train_data, val_data, test_data, source_unlabeled_data, target_unlabeled_data,source_vocab,target_vocab,vocab =cnn_load_data(source_domain,target_domain,data_path)
    
    source_term=feature_selection_df(source_vocab,5)
    target_term=feature_selection_df(target_vocab,5)
    pivot_term=set(source_term)&set(target_term)

    data= train_data+val_data+test_data+source_unlabeled_data+target_unlabeled_data
    source_data= train_data+val_data+source_unlabeled_data
    target_data = target_unlabeled_data
    print("selecting pivots...")
    pos_pivot, neg_pivot=filter_byWLLR(train_data,vocab,pivot_term,source_domain)
    max_story_size=max(map(len,(pairs[0] for pairs in data)))
    mean_story_size=int(np.mean([len(pairs[0]) for pairs in data]))
    sentences = map(len, (sentence for pairs in data for sentence in pairs[0]))
    max_sentence_size = max(sentences)
    mean_sentence_size = int(np.mean([len(sentence) for pairs in data for sentence in pairs[0]]))
    memory_size = min(args.memory_size, max_story_size)
    print("max  story size:", max_story_size)
    print("mean story size:", mean_story_size)
    print("max  sentence size:", max_sentence_size)
    print("mean sentence size:", mean_sentence_size)
    print("max memory size:", memory_size)
    max_sentence_size=args.sent_size

    word_embedding, word2idx, idx2word = get_w2vec(vocab, args)
    vocab_size = len(word_embedding)
    device = torch.device('cuda:0')
    x_train,  _,      y_train, u_train, v_train, word_mask_train = cnn_vectorize_data(train_data,  pos_pivot, neg_pivot, word2idx, 100)
    x_val,    _,      y_val,   u_val,   v_val,   word_mask_val   = cnn_vectorize_data(val_data,    pos_pivot, neg_pivot, word2idx, 100)
    x_test,   _,      y_test,  u_test,  v_test,  word_mask_test  = cnn_vectorize_data(test_data,   pos_pivot, neg_pivot, word2idx, 100)
    x_s,      d_s,    _,       u_s,     v_s,     word_mask_s     = cnn_vectorize_data(source_data, pos_pivot, neg_pivot, word2idx, 100)
    x_t,      d_t,    _,       u_t,     v_t,     word_mask_t     = cnn_vectorize_data(target_data, pos_pivot, neg_pivot, word2idx, 100)

    n_train  = x_train.size(0)
    n_test   = x_test.size(0)
    n_val    = x_val.size(0)
    n_source = x_s.size(0)
    n_target = x_t.size(0)
    n_domain = min(n_source, n_target)
    print(n_train, n_val, n_test, n_source, n_target)

    # for trainning
    x_train=torch.LongTensor(x_train).to(device)
    y_train=torch.LongTensor(y_train).to(device)
    u_train=torch.LongTensor(u_train).to(device)
    v_train=torch.LongTensor(v_train).to(device)
    word_mask_train=torch.LongTensor(word_mask_train).to(device)
 
    train_data = TensorDataset(x_train, y_train,u_train, v_train, word_mask_train)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
    

    x_val=torch.LongTensor(x_val).to(device)
    y_val=torch.LongTensor(y_val).to(device)
    u_val=torch.LongTensor(u_val).to(device)
    v_val=torch.LongTensor(v_val).to(device)
    word_mask_val=torch.LongTensor(word_mask_val).to(device)
    val_data = TensorDataset(x_val, y_val,u_val, v_val, word_mask_val)
    val_sampler = RandomSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=args.train_batch_size)

    x_test=torch.LongTensor(x_test).to(device)
    y_test=torch.LongTensor(y_test).to(device)
    u_test=torch.LongTensor(u_test).to(device)
    v_test=torch.LongTensor(v_test).to(device)
    word_mask_test=torch.LongTensor(word_mask_test).to(device)
    test_data = TensorDataset(x_test, y_test,u_test, v_test, word_mask_test)
    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.train_batch_size)

    x_d=torch.tensor(torch.cat([x_s,x_t],0)).to(device)
    y_d=torch.tensor(torch.cat([d_s,d_t],0)).to(device)
    u_d=torch.tensor(torch.cat([u_s,u_t],0)).to(device)
    v_d=torch.tensor(torch.cat([v_s,v_t],0)).to(device)
    word_mask_d=torch.tensor(torch.cat([word_mask_s,word_mask_t],0)).to(device)
    domain_data = TensorDataset(x_d, y_d,u_d,v_d,word_mask_d)
    domain_sampler = RandomSampler(domain_data)
    domain_dataloader = DataLoader(domain_data, sampler=domain_sampler, batch_size=args.domain_batch_size)


    
    model=CNN_SJ(args,word_embedding)
    model.word_embedding.weight.requires_grad=False
     
    # 初始化
    
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(),lr = args.lr,momentum = 0.9,weight_decay=0.005)
    # optimizer = torch.optim.Adam(model.parameters(),lr=0.005)
    

    train_steps = len(train_dataloader)
    domain_steps =len(domain_dataloader)
    val_steps =len(val_dataloader)
   
    train_iter =iter(train_dataloader)
    domain_iter =iter(domain_dataloader)
  
    best_val_loss= 1000000
    counter=0
    for i in range(args.num_train_epochs):
        model.train()
        p=float(i)/args.num_train_epochs
        adjust_learning_rate(optimizer, p)
        val_iter =iter(val_dataloader)
        to_loss=0.
        tr_loss=0.
        do_loss=0.
        uv_loss=0.
        for step in range(train_steps):
            try:
                batch = train_iter.next()
            except StopIteration:
                train_iter =iter(train_dataloader)
                batch = train_iter.next()
            
            xx_train, yy_train,uu_train, vv_train, wword_mask_train=batch
            au_xx_train=xx_train.mul(wword_mask_train)
            main_loss,tr_uv_loss1 = model(xx_train,au_xx_train,main_labels=yy_train,u_labels=uu_train,v_labels=vv_train,flag=0)
            try:
                batch = domain_iter.next()
            except StopIteration:
                domain_iter =iter(domain_dataloader)
                batch = domain_iter.next()

            x_d, y_d,u_d,v_d,word_mask_d=batch
            au_x_d=x_d.mul(word_mask_d)
            tr_uv_loss2= model(x_d,au_x_d,u_labels=u_d,v_labels=v_d,flag=1)
            loss=main_loss+tr_uv_loss1+tr_uv_loss2
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=40, norm_type=2)
            optimizer.step()
            to_loss += loss.item()
            tr_loss += main_loss.item()
        if i%args.val_interval==0:
            val_loss=0.
            vcorrect=0.
            vtotal=0.
            tcorrect=0.
            ttotal=0.
            with torch.no_grad():
                model.eval()
                train_iter =iter(train_dataloader)
                for step in range(train_steps):
                    batch = train_iter.next()
                
                    xx_train, yy_train,uu_train, vv_train, wword_mask_train=batch
                    au_xx_train=xx_train.mul(wword_mask_train)
                    logits = model(xx_train,au_xx_train,u_labels=uu_train,v_labels=vv_train,flag=2)
                    st=torch.nn.Softmax(-1)
                    logits=st(logits)
                    tcorrect+=(logits.argmax(dim = -1)==yy_train).sum()
                    ttotal += len(yy_train)
                for step in range(val_steps):
                    batch = val_iter.next()
                
                    x_val, y_val,u_val, v_val, word_mask_val=batch
                    au_x_val=x_val.mul(word_mask_val)
                    logits = model(x_val,au_x_val,u_labels=u_val,v_labels=v_val,flag=2)
                    
                    main_loss,au_loss= model(x_val,au_x_val,main_labels=y_val,u_labels=u_val,v_labels=v_val,flag=0)
                    val_loss+=main_loss.item()+c_l2_regularization_v(model,args.penalty).item()
                    st=torch.nn.Softmax(-1)
                    logits=st(logits)
                    vcorrect+=(logits.argmax(dim = -1)==y_val).sum()
                    vtotal += len(y_val)
                
                print("Eopch:%s lr:%.4f train_loss:%.2f total_loss:%.2f train_acc:%.2f val_loss:%.2f val_acc:%.2f"
                %(i,max(0.005/math.pow(1+10*p,0.75),0.002),tr_loss,to_loss,float(tcorrect)/ttotal,val_loss,float(vcorrect)/vtotal))

        if best_val_loss > val_loss:
            best_val_loss = val_loss
            best_epoch = i
            counter = 0
            torch.save(model, './work/models/'+source_domain+'_'+target_domain+'_cnn_au_best_model.pkl')
            
        else:
            counter += 1

        if counter == args.patience:
            break
            
        
    with torch.no_grad():
        model = torch.load('./work/models/'+source_domain+'_'+target_domain+'_cnn_au_best_model.pkl')
        model.to(device)
        model.eval()
        test_steps =len(test_dataloader)
        test_iter =iter(test_dataloader)
        
        correct=0.
        total=0.
        for step in range(test_steps):
            batch = test_iter.next()
            x_test, y_test,u_test, v_test, word_mask_test=batch
            au_x_test=x_test.mul(word_mask_test)
            logits = model(x_test,au_x_test,u_labels=u_test,v_labels=v_test,flag=2)
            st=torch.nn.Softmax(-1)
            logits=st(logits)
            correct+=(logits.argmax(dim = -1)==y_test).sum()
            total += len(y_test)
            
        print("Eopch:%s test_acc:%s"%(i,float(correct)/total))

        