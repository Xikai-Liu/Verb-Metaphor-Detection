import numpy as np
import gensim
class Worder(object):
    '''用于处理深度学习模型的输入
    1. 将句子中的词变为id
    2. 获取word2vec等词向量模型的weights，输入Embedding层中
    > 使用示例：  
    worder=Worder()
    worder.fit(train_sentences)
    worder.fit(valid_sentences)

    import matplotlib.pyplot as plt
    %matplotlib inline
    worder.plot_histogram(plt=plt)

    from gensim.models import Word2Vec
    word2vec_model=Word2Vec.load(word2vec_filename)
    worder.build(word2vec_model,word_num=40000)
    ids=worder.transform(train_sentences)
    
    3. 其中train_data的content为句子集合，每个句子中的词使用空格分割
    4. id转词  worder.vocabs
    5. 词转id  worder.vocab2id
    '''
    def __init__(self):
        self.prefix=['<pad>','<s>','<\s>','<unk>']  #特殊符号占位
        self.word_cnts={}   #词频统计
        self.total_cnt=0  #语料中总词数（不去重）
        self.vocabs=None
        self.weights=None
        
    def fit(self,sentences):
        for sentence in sentences:
            for word in sentence.lower().split(' '):
                word=word.strip()
                if len(word)==0:
                    continue
                self.word_cnts[word]=self.word_cnts.get(word,0)+1
                self.total_cnt+=1
        print('-- 当前总词数（不去重）:{0}, 去重总词数：{1}'.format(self.total_cnt,len(self.word_cnts)))
        
        
    def build_vocabs(self,word_num=40000):
        #构建词表
        word_num=word_num-len(self.prefix)
        tokens=sorted(self.word_cnts.items(),key=lambda x:x[1],reverse=True)
        tokens=tokens[:word_num]
        cnt=np.sum(token[1] for token in tokens)
        
        vocabs=self.prefix[:]+[token[0] for token in tokens]
        self.vocabs= vocabs
        self.vocab2id=dict([(token,i) for i,token in enumerate(vocabs)])
        
        print('-- 词表大小：{0}，词表覆盖度：{1}%'.format(len(self.vocabs),100*cnt/self.total_cnt))
        return vocabs
        
    def build_weights(self,keyed_vectors):
        '''keyed_vectors是gensim的KeyedVector对象,例如：
        model=Word2Vec.load(filename)
        keyed_vectors=model.wv
        '''
        #首先构建词表
        if self.vocabs is None:
            self.build_vocabs()
        
        if type(keyed_vectors) is gensim.models.word2vec.Word2Vec or \
           type(keyed_vectors) is gensim.models.fasttext.FastText:
            keyed_vectors=keyed_vectors.wv
        
        dim=keyed_vectors.vector_size
        
        weights=[np.random.uniform(low=-0.1,high=0.1,size=(dim,)) 
                 if word not in keyed_vectors 
                 else keyed_vectors[word] for word in self.vocabs]

        exist_cnt=len([word for word in self.vocabs if word in keyed_vectors])
        
        print('-- 词向量覆盖度： %d/%d, rate: %f%%'%(exist_cnt,len(self.vocabs),100*exist_cnt/len(self.vocabs)))
        self.weights=np.array(weights)
        
    def build(self,keyed_vectors=None,word_num=40000):
        self.build_vocabs(word_num)
        if keyed_vectors is not None:
            self.build_weights(keyed_vectors)
        print('词表及词向量构建完成!')
        
    def transform(self,sentences):
        data=[]
        unk=self.vocab2id['<unk>']
        for item in sentences:
            record=[self.vocab2id.get(word,unk) for word in item.lower().split(' ')]
            data.append(record)
        return data
    
    def clear(self):
        self.total_cnt=0
        self.word_cnts.clear()
        
    def plot_histogram(self,start_num=0,end_num=None, plt=None):
        '''绘制直方图，用来判断应该选择的词表大小
        '''
        tokens=sorted(self.word_cnts.items(),key=lambda x:x[1],reverse=True)
        if end_num is None:
            end_num=len(tokens)
        x=list(range(start_num,end_num,1000))
        x.append(end_num)
        y=[]
        k=0
        if x[k]==0:
            y.append(0)
            k+=1
        val=0
        for i,token in enumerate(tokens):
            val+=token[1]
            if k<len(x) and i+1==x[k]:
                k+=1
                y.append(val)
        y=np.array(y)/val
        if plt is not None:
            plt.grid('on')
            plt.plot(x,y)
        return x,y