import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import numpy as np

class SynonymModel(object):
    def __init__(self,filename,word2vec=None):
        # filename: './哈工大社会计算与信息检索研究中心同义词词林扩展版.txt'
        self.filenmae=filename
        self.word2vec=word2vec
        self.word2code=self.load_syno(filename)
        
    def load_syno(self,filename):
        word2code={}
        with open(filename,encoding='gbk') as f:
            for line in f:
                items=line.rstrip().split(' ')
                for i in range(1,len(items)):
                    if items[i] not in word2code:
                        word2code[items[i]]=[]
                    word2code[items[i]].append(items[0])
        print('载入词%d个'%len(word2code))
        return word2code
    
    def sample(self,words):
        i=np.random.randint(0,len(words))
        return words[i]
    
    def get_code(self,word,flag=None):
        if word in self.word2code:
            return self.sample(self.word2code[word])
        return word
    
synonym_model=SynonymModel('./哈工大社会计算与信息检索研究中心同义词词林扩展版.txt')
        
class MySentences(object):
    def __init__(self, filenames):
        self.filenames = filenames
 
    def __iter__(self):
        for fname in self.filenames:
            for line in open(fname,encoding='utf8'):
                words=line.lower().rstrip().split()
                yield [synonym_model.get_code(word).lower() for word in words]

if __name__=='__main__':
    model_name='word2vec_synonym_30'
    filenames=['/home/dutir923/yuml/datasets/smpcup2016/unlabeled_statuses/smp_all.txt',
               '/home/dutir923/yuml/datasets/stc_weibo2015/stc_weibo_train_post',
               '/home/dutir923/yuml/datasets/stc_weibo2015/stc_weibo_train_response'
              ]
    output_filename='/home/dutir923/yuml/datasets/word_embeddings/weibo20180310/'+model_name

    sentences=MySentences(filenames)
    if model_name.find('word2vec')>-1:
        model = gensim.models.Word2Vec(sentences, size=30,workers=10)
    else:
        model = gensim.models.FastText(sentences,size=620,workers=5)
    model.save(output_filename)
    logging.info('保存完毕')
