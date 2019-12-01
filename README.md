# rnn_project

------

简介：该项目是在IMDB电影数据集上的二分类模型，主要功能如下：

> * 1.模型训练集的准确率为90%左右，测试集80%左右
> * 2.抽取每句话的Input Vector,Hidden Vector和Output存为指定格式的文本文件

------

## 运行环境及配置
> * 1.Linux/macOS/Windows
> * 2.Python3
> * 3.Pytorch

------

## 代码及文件说明
> * all.npy：存储好的IMDB电影数据集5W条(正[1,25000]/负[25001,50000]各2.5w条)
> * wordVectors.npy：训练好的词向量，40w词，一个词50维
> * word_to_index.txt：词嵌入中每个词的下标与词的对应关系，如 0->i,1->am,2->a,3->student...
> * config.py：模型参数配置文件
> * pytorch.py：模型主要实现
> * tool.py：工具类
> * model_pytorch_gru：GRU存储模型的目录
> * model_pytorch_lstm：LSTM存储模型的目录


------

## 实现流程
Example:
> *                      词嵌入             LSMT/GRU
> * 一个句子：I am a student----->Input Vector-------->Result(Classification probability)

------

## 存储的文本内容格式说明

Sentence-[sentence-Index, word-Num]: I like playing football.
True-Label:positive
Predict-Label:positve
Predict-Prob:[negative-0.3,positive-0.7]
Word-[word-Index]-I: Embedding Input:[0.1,0.2,...,0.6]
Word-[word-Index]-I: Hidden State:[0.2,0.6,...,0.8]
Word-[word-Index]-I: Output :[0.3,0.7]
...

******************************************************************
Sentence-[sentence-Index, word-Num]:
...
...
...

------

## 输出说明
```python
在项目目录下生成pytorch_input_hidden_result_lstm.txt
```
------

## 程序运行相关
选用lstm并使用GPU:定位到配置文件config.py:根据需求修改model_used="gru",use_gpu = True即可
测试：定位到pytorch.py, 注释掉train(), 放开test()，训练同理


