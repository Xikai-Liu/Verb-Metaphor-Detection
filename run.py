import keras,os
from keras.engine.topology import Layer
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from metaphor_utils import AttentionPoolingLayer,ConcatLayer,SynonymModel
from metaphor_utils import ClearMaskLayer,ConvDense,MaskAveragePoolingLayer
from sklearn.metrics import roc_auc_score,accuracy_score,f1_score,precision_score,recall_score,log_loss
from keras.layers import LSTM,GRU,Bidirectional, TimeDistributed, Activation,concatenate
import pandas as pd
from keras.utils.np_utils import to_categorical
from gensim.models import Word2Vec
import jieba.posseg as psg
from keras.models import Model
from keras import layers
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping,ModelCheckpoint
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from preprocess import Worder
import pickle
from sklearn.model_selection import StratifiedShuffleSplit
import nltk

def init_gpu(gpu_id='0'):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES']=gpu_id
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    set_session(tf.Session(config=config))
    
'''
def psg_cut(contents):
    word_list=[]
    tag_list=[]
    for content in contents:
        words=[]
        tags=[]
        for token in psg.lcut(content):
            words.append(token.word)
            tags.append(token.flag)
        word_list.append(' '.join(words))
        tag_list.append(' '.join(tags))
    return word_list,tag_list
'''
def psg_cut(contents):
    word_list=[]
    tag_list=[]
    for content in contents:
        words=[]
        tags=[]
        lion=content.strip().split()
        pos_result = nltk.pos_tag(lion)
        #print(pos_result)
        #break
        for token in pos_result:
        #for token in psg.lcut(content):
            words.append(token[0])
            tags.append(token[1])
        word_list.append(' '.join(words))
        tag_list.append(' '.join(tags))
    return word_list,tag_list

def load_dataset(filename='./c.csv'):
    df=pd.read_csv(filename,sep='\t')
    df=df.fillna('')
    df['id']=df.index
    
    print('训练集',len(df[df.train_test=='train']))
    print('测试集',len(df[df.train_test=='test']))
    return df

def get_valid_test_idx():
    valid_data=[]
    test_data=[]
    for item in df.groupby('verb'):
        tmp_df=pd.DataFrame(item[1])
        negative_df=tmp_df[tmp_df.label==0].sample(frac=1.0)
        positive_df=tmp_df[tmp_df.label==1].sample(frac=1.0)
        if len(positive_df)>1 and len(negative_df)>1:
            if np.random.rand()>0.5:
                test_data.append(positive_df.index[0])
                test_data.append(negative_df.index[0])
            else:
                valid_data.append(positive_df.index[0])
                valid_data.append(negative_df.index[0])
    test_data=set(test_data)
    valid_data=set(valid_data)
    return valid_data,test_data

def generate_split_data(df,filename='./split2.pkl'):
    split_data=[]
    ids=df.id.tolist()
    for i in range(10):
        valid_idx,test_idx=get_valid_test_idx()
        train_idx=[id for id in ids if id not in valid_idx and id not in test_idx]
        split_data.append((train_idx,list(valid_idx),list(test_idx)))
    splitter=StratifiedShuffleSplit(train_size=0.1,test_size=0.1)
    for valid_idx,test_idx in splitter.split(df.id,df.label):
        valid_idx=set(valid_idx)
        test_idx=set(test_idx)
        train_idx=[id for id in ids if id not in valid_idx and id not in test_idx]
        split_data.append((train_idx,list(valid_idx),list(test_idx)))
    pickle.dump(split_data,open(filename,'wb'))
    
def load_split_data(filename):
    return pickle.load(open(filename,'rb'))

class ConcatLayer(Layer):
    def __init__(self,**kwargs):
        super(ConcatLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        title_xs,tag_x=inputs
        shape=title_xs.shape
        tx=K.expand_dims(tag_x,axis=-2)
        tx=K.tile(tx,[1,shape[1],1])
        outputs=concatenate([title_xs,tx],)
        return outputs

    def compute_output_shape(self, input_shape):
        return tuple(list(input_shape[0])[:-1]+[input_shape[0][-1]+input_shape[1][-1]])

    def compute_mask(self, inputs, mask=None):
        return mask[0]


def get_abstract_attnn_model(word_worder,tag_worder,code_worder):
    word_input=layers.Input(shape=(19,))
    tag_input=layers.Input(shape=(19,))

    verb_input=layers.Input(shape=(None,))
    subject_input=layers.Input(shape=(None,))
    object_input=layers.Input(shape=(None,))
    abstract_inputs=[verb_input,subject_input,object_input]

    #原始词向量
    word_emb=layers.Embedding(input_dim=word_worder.weights.shape[0],
                              output_dim=word_worder.weights.shape[1],
                            weights=[word_worder.weights],mask_zero=True)(word_input)
    #词性向量
    tag_emb=layers.Embedding(input_dim=len(tag_worder.vocabs),
                             output_dim=10,
                             mask_zero=True)(tag_input)

    #同义词林向量
    synonym_emb_layer=layers.Embedding(input_dim=len(code_worder.vocabs),
                             output_dim=code_worder.weights.shape[1],
                            mask_zero=True,weights=[code_worder.weights])
    abstract_embs=[]
    for input in abstract_inputs:
        xs=synonym_emb_layer(input)
        emb=MaskAveragePoolingLayer()(xs)
        abstract_embs.append(emb)

    abstract_emb=concatenate(abstract_embs)
    abstract_emb=layers.Dense(30,activation='tanh')(abstract_emb)
    xs=concatenate([word_emb,tag_emb,])

    xs=ConcatLayer()([xs,abstract_emb])

    xs=ConvDense(128,activation='relu')(xs)
    xs=Bidirectional(layers.LSTM(128,return_sequences=True))(xs)
    xs=layers.Dropout(0.5)(xs)
    xs=AttentionPoolingLayer()(xs)

    xs=layers.Dropout(0.5)(xs)
    xs=layers.Dense(100,activation='tanh')(xs)
    dense_xs=layers.Dropout(0.5)(xs)

    output=layers.Dense(1,activation='sigmoid')(dense_xs)

    model=Model(inputs=[word_input,tag_input,verb_input,subject_input,object_input],outputs=[output])
    model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
    return model

def get_tag_attnn_model(word_worder,tag_worder,code_worder):
    word_input=layers.Input(shape=(19,))
    tag_input=layers.Input(shape=(19,))

    verb_input=layers.Input(shape=(None,))
    subject_input=layers.Input(shape=(None,))
    object_input=layers.Input(shape=(None,))
    abstract_inputs=[verb_input,subject_input,object_input]

    #原始词向量
    word_emb=layers.Embedding(input_dim=word_worder.weights.shape[0],
                              output_dim=word_worder.weights.shape[1],
                            weights=[word_worder.weights],mask_zero=True)(word_input)
    #词性向量
    tag_emb=layers.Embedding(input_dim=len(tag_worder.vocabs),
                             output_dim=10,
                             mask_zero=True)(tag_input)


    xs=concatenate([word_emb,tag_emb,])

    #xs=ConvDense(128,activation='relu')(xs)
    xs=layers.LSTM(128,return_sequences=True)(xs)
    xs=layers.Dropout(0.5)(xs)
    xs=AttentionPoolingLayer()(xs)

    xs=layers.Dropout(0.5)(xs)
    xs=layers.Dense(100,activation='tanh')(xs)
    dense_xs=layers.Dropout(0.5)(xs)

    output=layers.Dense(1,activation='sigmoid')(dense_xs)

    model=Model(inputs=[word_input,tag_input,verb_input,subject_input,object_input],outputs=[output])
    model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
    return model

def get_cnn_model(word_worder,tag_worder,code_worder):
    word_input=layers.Input(shape=(19,))
    tag_input=layers.Input(shape=(19,))

    verb_input=layers.Input(shape=(None,))
    subject_input=layers.Input(shape=(None,))
    object_input=layers.Input(shape=(None,))
    abstract_inputs=[verb_input,subject_input,object_input]

    #原始词向量
    word_emb=layers.Embedding(input_dim=word_worder.weights.shape[0],
                              output_dim=word_worder.weights.shape[1],
                            weights=[word_worder.weights],mask_zero=False)(word_input)


    kernel_sizes=[2,3]
    conv_xs_list=[]
    for kernel_size in kernel_sizes:
        tmp_conv_xs=layers.Convolution1D(256,kernel_size,activation='relu')(word_emb)
        tmp_conv_xs=layers.Dropout(0.3)(tmp_conv_xs)
        tmp_conv_xs=layers.GlobalMaxPool1D()(tmp_conv_xs)
        conv_xs_list.append(tmp_conv_xs)
        
    xs=concatenate(conv_xs_list)

    xs=layers.Dropout(0.5)(xs)
    xs=layers.Dense(100,activation='tanh')(xs)
    dense_xs=layers.Dropout(0.5)(xs)

    output=layers.Dense(1,activation='sigmoid')(dense_xs)

    model=Model(inputs=[word_input,tag_input,verb_input,subject_input,object_input],outputs=[output])
    model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
    return model

def get_lstm_model(word_worder,tag_worder,code_worder):
    word_input=layers.Input(shape=(19,))
    tag_input=layers.Input(shape=(19,))

    verb_input=layers.Input(shape=(None,))
    subject_input=layers.Input(shape=(None,))
    object_input=layers.Input(shape=(None,))
    abstract_inputs=[verb_input,subject_input,object_input]

    #原始词向量
    word_emb=layers.Embedding(input_dim=word_worder.weights.shape[0],
                              output_dim=word_worder.weights.shape[1],
                            weights=[word_worder.weights],mask_zero=True)(word_input)
    #词性向量
    tag_emb=layers.Embedding(input_dim=len(tag_worder.vocabs),
                             output_dim=10,
                             mask_zero=True)(tag_input)

    #同义词林向量
    synonym_emb_layer=layers.Embedding(input_dim=len(code_worder.vocabs),
                             output_dim=code_worder.weights.shape[1],
                            mask_zero=True,weights=[code_worder.weights])
   


    xs=layers.Bidirectional(layers.LSTM(128,))(word_emb)
   
    xs=layers.Dropout(0.5)(xs)
    print(xs.shape)
    xs=layers.Dense(100,activation='tanh')(xs)
    dense_xs=layers.Dropout(0.5)(xs)
    print(dense_xs.shape)
    output=layers.Dense(1,activation='sigmoid')(dense_xs)

    model=Model(inputs=[word_input,tag_input,verb_input,subject_input,object_input],outputs=[output])
    model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
    return model

def get_lstm_att_model(word_worder,tag_worder,code_worder):
    word_input=layers.Input(shape=(19,))
    tag_input=layers.Input(shape=(19,))

    verb_input=layers.Input(shape=(None,))
    subject_input=layers.Input(shape=(None,))
    object_input=layers.Input(shape=(None,))
    abstract_inputs=[verb_input,subject_input,object_input]

    #原始词向量
    word_emb=layers.Embedding(input_dim=word_worder.weights.shape[0],
                              output_dim=word_worder.weights.shape[1],
                            weights=[word_worder.weights],mask_zero=True)(word_input)
    #词性向量
    tag_emb=layers.Embedding(input_dim=len(tag_worder.vocabs),
                             output_dim=10,
                             mask_zero=True)(tag_input)

    #同义词林向量
    synonym_emb_layer=layers.Embedding(input_dim=len(code_worder.vocabs),
                             output_dim=code_worder.weights.shape[1],
                            mask_zero=True,weights=[code_worder.weights])
   

    xs=layers.LSTM(128,return_sequences=True)(word_emb)
    xs=layers.Dropout(0.5)(xs)
    xs=AttentionPoolingLayer()(xs)
   
    xs=layers.Dropout(0.5)(xs)
    xs=layers.Dense(100,activation='tanh')(xs)
    dense_xs=layers.Dropout(0.5)(xs)

    output=layers.Dense(1,activation='sigmoid')(dense_xs)

    model=Model(inputs=[word_input,tag_input,verb_input,subject_input,object_input],outputs=[output])
    model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
    return model

def train(model,train_data,train_ys,valid_data,valid_ys,model_filename,):
    print('正在训练...',model_filename)
    model_checkpoint=ModelCheckpoint(model_filename,save_best_only=True,)
    early_stop_callback=EarlyStopping(patience=50,monitor='val_loss')
    model.fit(train_data,train_ys,batch_size=512,
              validation_data=[valid_data,valid_ys],
              epochs=300,verbose=0,
              callbacks=[early_stop_callback,model_checkpoint])
    model.load_weights(model_filename)

def test(model,test_data,test_ys,thres=0.5):
    yp=model.predict(test_data)[:,0]
    #loss=log_loss(test_ys[:,0],yp)
    #auc=roc_auc_score(test_ys[:,0],yp)
    #acc=accuracy_score(test_ys[:,0],yp>thres)
    #p=precision_score(test_ys[:,0],yp>thres)
    #r=recall_score(test_ys[:,0],yp>thres)
    #f1=f1_score(test_ys[:,0],yp>thres)
    loss=log_loss(test_ys,yp)
    auc=roc_auc_score(test_ys,yp)
    acc=accuracy_score(test_ys,yp>thres)
    p=precision_score(test_ys,yp>thres)
    r=recall_score(test_ys,yp>thres)
    f1=f1_score(test_ys,yp>thres)
    print('loss:\t',loss)
    print('auc:\t',auc)
    print('acc:\t',acc)
    print('precision:\t',p)
    print('recall:\t',r)
    print('f1:\t',f1)
    print('--------')
    return [loss,auc,acc,p,r,f1]

def run_model(model_name,get_model_func):
    
    result_data=[]
    cnt=1
    for train_idx,valid_idx,test_idx in split_data:
        '''开始训练'''
        train_data=[feature[train_idx] for feature in data]
        valid_data=[feature[valid_idx] for feature in data]
        test_data=[feature[test_idx] for feature in data]
        #train_ys=to_categorical(np.asarray((df.label.values[train_idx])))
        #valid_ys=to_categorical(np.asarray((df.label.values[valid_idx])))
        #test_ys=to_categorical(np.asarray((df.label.values[test_idx])))
        valid_ys=df.label.values[valid_idx]
        test_ys=df.label.values[test_idx]
        train_ys=df.label.values[train_idx]
        model=get_model_func(word_worder,tag_worder,code_worder)
        train(model,train_data,train_ys,valid_data,valid_ys,os.path.join(model_path,'%s_%d'%(model_name,cnt)))
        valid_result=test(model,valid_data,valid_ys)
        test_result=test(model,test_data,test_ys)
        result_data.append(valid_result+test_result)
        cnt+=1

    result_df=pd.DataFrame(result_data,columns=headers)
    result_df.to_csv(os.path.join(results_path,'%s.csv'%model_name),index=False)
    print(model_name,'训练完毕')

if __name__=='__main__':
    init_gpu('0')  #配置gpu
    
    model_path='./models02'
    results_path='./results02'
    w2v_model=Word2Vec.load('wiki.en.text.model')
    code_w2v_model=Word2Vec.load('word2vec_synonym_30')
    synonym_model=SynonymModel('./Total.txt')
    split_data=load_split_data('./splitt.pkl')  #保存了训练集验证集和测试集各自的id
    
    if os.path.exists(model_path)==False: os.mkdir(model_path)
    if os.path.exists(results_path)==False: os.mkdir(results_path)

    '''大写变小写'''
    code_keys=list(code_w2v_model.wv.vocab.keys())
    for v in code_keys:
        code_w2v_model.wv.vocab[v.lower()]=code_w2v_model.wv.vocab[v]
    
    '''语料预处理'''
    df=load_dataset()
    df['verb_code']=df.verb.apply(lambda x:' '.join(synonym_model.get_codes(x)))
    df['subject_code']=df.subject.apply(lambda x:' '.join(synonym_model.get_codes(x)))
    df['object_code']=df.object.apply(lambda x:' '.join(synonym_model.get_codes(x)))
    word_list,tag_list=psg_cut(df.content.tolist())  #分词及词性分析

    '''特征计算'''
    word_worder=Worder()
    word_worder.fit(word_list)
    word_worder.build(w2v_model)
    word_ids=word_worder.transform(word_list)

    tag_worder=Worder()
    tag_worder.fit(tag_list)
    tag_worder.build()
    tag_ids=tag_worder.transform(tag_list)

    code_worder=Worder()
    code_worder.fit(df.verb_code.tolist())
    code_worder.fit(df.subject_code.tolist())
    code_worder.fit(df.object_code.tolist())
    code_worder.build(code_w2v_model)

    verb_ids=code_worder.transform(df.verb_code.tolist())
    subject_ids=code_worder.transform(df.subject_code.tolist())
    object_ids=code_worder.transform(df.object_code.tolist())

    word_ids=pad_sequences(word_ids,padding='post')
    tag_ids=pad_sequences(tag_ids,padding='post')
    verb_ids=pad_sequences(verb_ids,padding='post')
    subject_ids=pad_sequences(subject_ids,padding='post')
    object_ids=pad_sequences(object_ids,padding='post')
    
    data=[word_ids,tag_ids,verb_ids,subject_ids,object_ids]

    headers='valid_loss,valid_auc,valid_acc,valid_prec,valid_recall,valid_f1,'
    headers=headers+'test_loss,test_auc,test_acc,test_prec,test_recall,test_f1'
    headers=headers.split(',')
    print('数据处理完毕')
    
    '''训练模型'''
    print('运行 abstract_attnn')
    run_model('abstract_attnn',get_abstract_attnn_model)
    #run_model('abstract_attnn',get_abstract_attnn_model)
    print('运行 cnn模型')
    run_model('cnn',get_cnn_model)
    print('运行 lstm模型')
    run_model('lstm',get_lstm_model)
    print('运行 lstm_att模型')
    run_model('lstm_att',get_lstm_att_model)
    print('运行 tag_att 模型')
    run_model('tag_att',get_tag_attnn_model)
    
    
    '''生成报告'''
    models_names=['cnn','lstm','lstm_att','tag_att','abstract_attnn']
    results=[pd.read_csv(os.path.join(results_path,'%s.csv'%name)) for name in models_names]
    total_df=pd.DataFrame([result.mean() for result in results])
    total_df.insert(0,'name',models_names)
    total_df.to_csv(os.path.join(results_path,'total.csv'),index=False)
    print('运行结束')