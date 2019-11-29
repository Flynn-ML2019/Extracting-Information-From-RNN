#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from config import DefaultConfig
from tool import Tools  
class SentimentNet(nn.Module): 
    def __init__(self,  embed_size, num_hiddens, num_layers,
                 bidirectional, weight, labels, use_gpu,model_select, **kwargs):
        super(SentimentNet, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.use_gpu = use_gpu
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(opt.word_embed_size, embed_size)
        self.embedding.weight.data.copy_(torch.from_numpy(opt.wordVectors))
        self.embedding.weight.requires_grad = False
        if opt.model_used=="gru":
            self.encoder = nn.GRU(input_size=embed_size, hidden_size=self.num_hiddens,
                               num_layers=num_layers, bidirectional=self.bidirectional,batch_first=True)
        elif opt.model_used=="lstm":
            self.encoder = nn.LSTM(input_size=embed_size, hidden_size=self.num_hiddens,
                               num_layers=num_layers, bidirectional=self.bidirectional,batch_first=True)
        if self.bidirectional:
            self.decoder = nn.Linear(num_hiddens * 2, labels)
        else:
            self.decoder = nn.Linear(num_hiddens * 1, labels)
    def get_Squence_from_indexs(self,word_indexs,lengthOfWord):
        word_index={}
        f=open("./word_to_index.txt","r")
        line=f.readline()
        while line:
            word_index[int(line.split()[0])]=line.split()[1]
            line=f.readline()
        result=[]
        for i in range(len(word_indexs)):
            if word_indexs[i]!=0:
                result.append(word_index.get(word_indexs[i])+" ")
        return result
    def forward(self, inputs,label,flag):
        global sequence_count
        embeddings = self.embedding(inputs) #嵌入后的输入
        out, hidden = self.encoder(embeddings)
        encoding = out[:,-1,:]
        outputs = self.decoder(encoding)
        
        sen_word_length=[] #获取每个句子的真实长度
        for x in range(opt.batch_size_train):
            count=0
            for y in range(opt.maxSelength):
                if inputs[x][y]!=0:
                    count=count+1
                else:
                    break
            sen_word_length.append(count)
        f=open("./pytorch_input_hidden_result_"+opt.model_used+"_"+flag+".txt","a")
        for i in range(opt.batch_size_train):
            for j in range(sen_word_length[i]):
                input1=embeddings.numpy()[i][j]  #第i个句子的第j个单词的输入
                hidden1=out.detach().numpy()[i][j] #第i个句子的第j个单词的hidden
                result=self.decoder(out[i][j])     #第i个句子的第j个单词的输出概率
                result=result.view(1,len(result))
                result=F.softmax(result,dim=1)
                all_words=self.get_Squence_from_indexs(inputs.numpy()[i],sen_word_length[i]+1)
                f.write("Sentence-["+str(opt.sequence_count)+","+str(sen_word_length[i])\
                    +"]:"+''.join(all_words)+"\n")
                sequence_isPos="negative"
                if all(label.numpy()[i]==[1,0]):
                    sequence_isPos="positive"
                f.write("True-Label:"+sequence_isPos+"\n")
                result1=self.decoder(out[i][sen_word_length[i]-1])     #第i个句子的最后的输出概率
                result1=result1.view(1,len(result1))
                result1=F.softmax(result1,dim=1)
                neg=result1.detach().numpy()[0][0]
                pos=result1.detach().numpy()[0][1]
                sequence_Pre_Pos="positive"
                if neg>=pos:
                    sequence_Pre_Pos="negative"
                f.write("Predict-Label:"+sequence_Pre_Pos+"\n")
                f.write("Predict-Prob:[negative-"+str(neg)+",positive-"+str(pos)+"]\n")
                f.write("Word-["+str(j+1)+"]-"+all_words[j]+": Embedding Input:"+str(input1)+"\n")
                f.write("Word-["+str(j+1)+"]-"+all_words[j]+": Hidden State:"+str(hidden1)+"\n")
                f.write("Word-["+str(j+1)+"]-"+all_words[j]+": Output:"+str(result.detach().numpy()[0])+"\n")
                for k in range(j+1,sen_word_length[i]):
                    input1=embeddings.numpy()[i][k]  #第i个句子的第j个单词的输入
                    hidden1=out.detach().numpy()[i][k] #第i个句子的第j个单词的hidden
                    result=self.decoder(out[i][k])     #第i个句子的第j个单词的输出概率
                    result=result.view(1,len(result))
                    result=F.softmax(result,dim=1)
                    f.write("Word-["+str(k+1)+"]-"+all_words[k]+": Embedding Input:"+str(input1)+"\n")
                    f.write("Word-["+str(k+1)+"]-"+all_words[k]+": Hidden State:"+str(hidden1)+"\n")
                    f.write("Word-["+str(k+1)+"]-"+all_words[k]+": Output:"+str(result.detach().numpy()[0])+"\n")
                f.write("********************************************************************************************************"+"\n")
                f.flush()
                opt.sequence_count=opt.sequence_count+1
                break
            
        return outputs

def train(): #训练
    net = SentimentNet(embed_size=opt.embed_size,
                       num_hiddens=opt.num_hiddens,
                       num_layers=opt.num_layers,
                       bidirectional=opt.bidirectional,
                       weight=opt.wordVectors,
                       labels=opt.labels, use_gpu=opt.use_gpu,model_select=opt.model_used)
    loss_function = nn.CrossEntropyLoss( )

    #optimizer = optim.SGD(net.parameters(), lr=lr)
    optimizer =optim.Adam(net.parameters(), lr=opt.lr)
    #optimizer=optim.SGD(net.parameters(), lr=lr, momentum=0.8)
    for epoch in range(opt.num_epochs):
        train_loss= 0
        train_acc= 0
        n=0
        for i,(feature, label) in enumerate(trainloader):
            n=n+1
            net.zero_grad()
            feature = Variable(feature)
            label = Variable(label)
            score = net(feature,label,"train")
            #feature=feature.cuda()
            #label=label.cuda()
            loss = loss_function(score,torch.max(label, 1)[1])
            loss.backward()
            optimizer.step()
            acc=accuracy_score(torch.argmax(score.cpu().data,
                                                      dim=1), torch.max(label, 1)[1].cpu())
            train_acc += acc
            train_loss += loss

            print('epoch: %d, train loss: %.4f, train acc avg: %.2f ,  acc: %.2f ' %
                  (epoch, train_loss.data / n, (train_acc / n),acc))

        with torch.no_grad():
            m=0
            test_acc= 0
            for i,(feature, label) in enumerate(testloader):
                m=m+1
                feature = Variable(feature)
                label = Variable(label)
                #feature=feature.cuda()
                #label=label.cuda()
                score = net(feature,label,"test")
                train = accuracy_score(torch.argmax(score.cpu().data,
                                                                 dim=1), torch.max(label, 1)[1].cpu())
                test_acc= test_acc+train
            print('   test acc: %.2f ' %(test_acc/m))
        if opt.model_used=="lstm":
            
            torch.save(net, './model_pytorch_lstm/model.pkl')
        elif opt.model_used=="gru":
            torch.save(net, './model_pytorch_gru/model.pkl')


def test(): #测试
    if opt.model_used=="lstm":
        net_trained = torch.load('./model_pytorch_lstm/model.pkl')
    elif opt.model_used=="gru":
        net_trained = torch.load('./model_pytorch_gru/model.pkl')
    test_acc=0
    m=1
    for i,(feature, label) in enumerate(testloader):
        net_trained.zero_grad()
        feature = Variable(feature)
        label = Variable(label)
        #feature=feature.cuda()
        #label=label.cuda()
        score = net_trained(feature,label,"test")
        train = accuracy_score(torch.argmax(score.cpu().data,
                                                 dim=1), torch.max(label, 1)[1].cpu())
        test_acc= test_acc+train
        m=m+1
    print('test acc: %.2f ' %(test_acc/m))

if __name__ == "__main__":                
    opt = DefaultConfig() #加载超参数和相关配置  
             
    tools = Tools() #
         
    inputs_,lables_=tools.get_data(opt) #加载数据   

    trainloader,testloader=tools.split_data(opt,inputs_,lables_) #划分训练 测试数据

    #train() #训练模型
     
    test()  #测试