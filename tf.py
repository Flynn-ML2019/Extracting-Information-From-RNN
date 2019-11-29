#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from random import randint
from os.path import isfile, join
import tensorflow as tf

np.set_printoptions(threshold = np.inf) 
np.set_printoptions(suppress = True)
gru_lstm_switch="lstm"
sequence_count=1
#gru_lstm_switch="gru"
class extract_data:
    #该矩阵由 GloVe 进行训练得到。矩阵将包含 400000 个词向量，每个向量的维数为 50
    wordVectors = np.load('wordVectors.npy')
    wordsList = np.load('wordsList.npy')
    wordsList = wordsList.tolist() 
    wordsList = [word.decode('UTF-8') for word in wordsList]
    max_seq_num = 250 #句子最大长度
    num_dimensions = 50 #单词嵌入维度
    batch_size = 24 
    lstm_units = 64 
    num_labels = 2  
    ids = np.load('idsMatrix.npy')    
    tf.reset_default_graph()

    labels = tf.placeholder(tf.float32, [batch_size, num_labels],name="labels")
    input_data = tf.placeholder(tf.int32, [batch_size, max_seq_num],name="input_data")

    data = tf.Variable(tf.zeros([batch_size, max_seq_num, num_dimensions]), dtype=tf.float32)
    data = tf.nn.embedding_lookup(wordVectors, input_data,name="data")
    if gru_lstm_switch=="gru":
        gruCell = tf.nn.rnn_cell.GRUCell(lstm_units)
        value, states = tf.nn.dynamic_rnn(gruCell, data, dtype=tf.float32)
        print("model use:gru")
    elif gru_lstm_switch=="lstm":
        lstmCell = tf.nn.rnn_cell.LSTMCell(lstm_units)
        value, states = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)
        print("model use:lstm")

    keep_value=value   
    weight = tf.Variable(tf.truncated_normal([lstm_units, num_labels]))
    bias = tf.Variable(tf.constant(0.1, shape=[num_labels]))


    x=tf.placeholder(shape=[],dtype=tf.int64,name="x")
    y=tf.placeholder(shape=[],dtype=tf.int64,name="y")
    temp=tf.reshape(value[x][y],(1,64)) #第x句话的第y个单词的概率
    want = (tf.matmul(temp, weight) + bias) 
    hh=tf.nn.softmax(want,axis=1,name="hh")

    #可舍弃 每句话最后的概率
    value = tf.transpose(value, [1, 0, 2])
    last = tf.gather(value, int(value.get_shape()[0]) - 1) 
    prediction = (tf.matmul(last, weight) + bias)
    result_want_one=tf.nn.softmax(prediction,axis=1,name="result_want_one") 

    correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=labels))
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()   
    sess.run(tf.global_variables_initializer())
    #获取训练数据
    def get_train_batch(self):
        labels = []      
        arr = np.zeros([self.batch_size, self.max_seq_num])
        for i in range(0,self.batch_size):       
            if (i % 2 == 0):
                num = randint(1, 11499)
                labels.append([1, 0])
            else:
                num = randint(13499, 24999)
                labels.append([0, 1])                                                 
            arr[i] = self.ids[num - 1:num]
        return arr, labels
    #获取测试数据
    def get_test_batch(self): 
        labels = []
        arr = np.zeros([self.batch_size, self.max_seq_num])
        word_kength=[]
        for i in range(0,self.batch_size):
            num = randint(11499, 13499) 
            if (num <= 12499):
                labels.append([1, 0])
            else:
                labels.append([0, 1])
            arr[i] = self.ids[num - 1:num]    
            count=0
            for j in range(246):      
                #确保一句话的单词的完整性                        
                if arr[i][j]!=0:
                    count=count+1    
            word_kength.append(count)
        return arr,labels,word_kength 
    def get_Squence_from_indexs(self,word_indexs):
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
    def test(self):
        global sequence_count
        with tf.Session() as sess:
            iterations = 10 #一次迭代24个句子
            count = 1       
            for x in range(iterations):
                next_batch, next_batch_labels,sen_word_length = self.get_test_batch()
                #自己指定加载的模型
                saver = tf.train.import_meta_graph('./model_tf_'+gru_lstm_switch+'/pretrained_lstm.ckpt-4000.meta')
                saver.restore(sess,tf.train.latest_checkpoint('./model_tf_'+gru_lstm_switch))
                graph = tf.get_default_graph()              
                f=open("./tf_input_hidden_result_"+gru_lstm_switch+"_test.txt","a")       
                for i in range(self.batch_size): 
                    for k in range(sen_word_length[i]):
                        all_words=self.get_Squence_from_indexs(next_batch[i])
                        f.write("Sentence-["+str(sequence_count)+","+str(sen_word_length[i])\
                            +"]:"+''.join(all_words)+"\n")
                        sequence_isPos="negative"
                        if next_batch_labels[i][0]==1:       
                            sequence_isPos="positive"           
                        f.write("True-Label:"+sequence_isPos+"\n")         
                        outputs=sess.run(self.hh,feed_dict={self.x: i, self.y:sen_word_length[i]-1 ,self.input_data: next_batch, self.labels: next_batch_labels})
                                                       
                        pos=outputs[0][1]
                        neg=outputs[0][0]       
                        sequence_Pre_Pos="positive"                
                        if neg>=pos:
                            sequence_Pre_Pos="negative"
                        f.write("Predict-Label:"+sequence_Pre_Pos+"\n")
                        f.write("Predict-Prob:[negative-"+str(neg)+",positive-"+str(pos)+"]\n")
                        input1=sess.run(self.data,feed_dict={self.input_data: next_batch, self.labels: next_batch_labels}) 
                        hidden1=sess.run(self.keep_value,feed_dict={self.input_data: next_batch, self.labels: next_batch_labels})   
                        result=sess.run(self.hh,feed_dict={self.x: i, self.y: k,self.input_data: next_batch, self.labels: next_batch_labels})
                        f.write("Word-["+str(k+1)+"]-"+all_words[k]+": Embedding Input:"+str(input1[i][k])+"\n")
                        f.write("Word-["+str(k+1)+"]-"+all_words[k]+": Hidden State:"+str(hidden1[i][k])+"\n")
                        f.write("Word-["+str(k+1)+"]-"+all_words[k]+": Output:"+str(result[0])+"\n")
                        for j in range(k+1,sen_word_length[i]):                                          
                            result=sess.run(self.hh,feed_dict={self.x: i, self.y: j,self.input_data: next_batch, self.labels: next_batch_labels})
                            f.write("Word-["+str(j+1)+"]-"+all_words[j]+": Embedding Input:"+str(input1[i][j])+"\n")
                            f.write("Word-["+str(j+1)+"]-"+all_words[j]+": Hidden State:"+str(hidden1[i][j])+"\n")
                            f.write("Word-["+str(j+1)+"]-"+all_words[j]+": Output:"+str(result[0])+"\n")
                        f.write("********************************************************************************************************"+"\n")
                        f.flush()        
                        sequence_count=sequence_count+1
                        f.flush() 
                        break
                    print("第",count,"组句子抽取完毕. ",count,"/",iterations*self.batch_size)
                    count = count+1
                f.close()

    def train(self):
        global sequence_count
        iterations = 20001 #总迭代次数
        count = 1 
        with tf.Session() as sess:
	        for i in range(iterations):      
	            #print("\r ",(i/iterations)*100,"%",end="") 
	            next_batch, next_batch_labels,sen_word_length = self.get_test_batch()
	            #自己指定加载的模型
	            saver = tf.train.import_meta_graph('./model_tf_'+gru_lstm_switch+'/pretrained_lstm.ckpt-4000.meta')
	            saver.restore(sess,tf.train.latest_checkpoint('./model_tf_'+gru_lstm_switch))
	            graph = tf.get_default_graph()              
	            f=open("./tf_input_hidden_result_"+gru_lstm_switch+"_train.txt","a")       
	            for i in range(self.batch_size): 
	                for k in range(sen_word_length[i]):
	                    all_words=self.get_Squence_from_indexs(next_batch[i])
	                    f.write("Sentence-["+str(sequence_count)+","+str(sen_word_length[i])\
	                        +"]:"+''.join(all_words)+"\n")
	                    sequence_isPos="negative"
	                    if next_batch_labels[i][0]==1:           
	                        sequence_isPos="positive"           
	                    f.write("True-Label:"+sequence_isPos+"\n")         
	                    outputs=sess.run(self.hh,feed_dict={self.x: i, self.y:sen_word_length[i]-1 ,self.input_data: next_batch, self.labels: next_batch_labels})
	                                                   
	                    pos=outputs[0][1]
	                    neg=outputs[0][0]       
	                    sequence_Pre_Pos="positive"                
	                    if neg>=pos:
	                        sequence_Pre_Pos="negative"
	                    f.write("Predict-Label:"+sequence_Pre_Pos+"\n")
	                    f.write("Predict-Prob:[negative-"+str(neg)+",positive-"+str(pos)+"]\n")
	                    input1=sess.run(self.data,feed_dict={self.input_data: next_batch, self.labels: next_batch_labels}) 
	                    hidden1=sess.run(self.keep_value,feed_dict={self.input_data: next_batch, self.labels: next_batch_labels})   
	                    result=sess.run(self.hh,feed_dict={self.x: i, self.y: k,self.input_data: next_batch, self.labels: next_batch_labels})
	                    f.write("Word-["+str(k+1)+"]-"+all_words[k]+": Embedding Input:"+str(input1[i][k])+"\n")
	                    f.write("Word-["+str(k+1)+"]-"+all_words[k]+": Hidden State:"+str(hidden1[i][k])+"\n")
	                    f.write("Word-["+str(k+1)+"]-"+all_words[k]+": Output:"+str(result[0])+"\n")
	                    for j in range(k+1,sen_word_length[i]):                                          
	                        result=sess.run(self.hh,feed_dict={self.x: i, self.y: j,self.input_data: next_batch, self.labels: next_batch_labels})
	                        f.write("Word-["+str(j+1)+"]-"+all_words[j]+": Embedding Input:"+str(input1[i][j])+"\n")
	                        f.write("Word-["+str(j+1)+"]-"+all_words[j]+": Hidden State:"+str(hidden1[i][j])+"\n")
	                        f.write("Word-["+str(j+1)+"]-"+all_words[j]+": Output:"+str(result[0])+"\n")
	                    f.write("********************************************************************************************************"+"\n")
	                    f.flush()        
	                    sequence_count=sequence_count+1
	                    f.flush() 
	                    break
	                print("第",count,"组句子测试阶段抽取完毕. ",count,"/",iterations*self.batch_size)
	                count = count+1
	            f.close()

	            next_batch, next_batch_labels = self.get_train_batch()
	            self.sess.run(self.optimizer,{self.input_data: next_batch, self.labels: next_batch_labels})                          
	            loss_ = self.sess.run(self.loss, {self.input_data: next_batch, self.labels: next_batch_labels})
	            accuracy_=(self.sess.run(self.accuracy, {self.input_data: next_batch, self.labels: next_batch_labels})) * 100
	            print("\r 迭代次数:{}/{} loss:{} accuracy:{}".format(i+1, iterations,loss_,accuracy_),end="") 
	            #1000次保存一下模型    1w次可达70-80准确率
	            if (i % 1000 == 0 and i != 0):
	                save_path = self.saver.save(self.sess, "./model_tf_"+gru_lstm_switch+"/pretrained_lstm.ckpt", global_step=i)
	                print("saved to %s" % save_path) 

l = extract_data()
#l.train()
l.test()

