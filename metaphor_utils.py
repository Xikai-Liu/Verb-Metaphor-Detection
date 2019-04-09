import numpy as np
from keras import backend as K
from keras.engine.topology import Layer
from keras import activations, initializers, regularizers, constraints
from keras.utils import conv_utils
from keras.engine import InputSpec
from keras.layers.merge import concatenate

class SynonymModel(object):
    '''同义词词林
    调用 get_codes函数，传入一个单词，返回其所在的所有树节点编号
    '''
    def __init__(self,filename):
        # filename: './hgdtyc.txt'
        self.filenmae=filename
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
    
    def get_codes(self,word,sample=False):
        if word in self.word2code:
            codes=self.word2code[word]
            if sample:
                return [self.sample(codes)]
            else:
                return codes
        return ['']

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
    
class AttentionPoolingLayer(Layer):
    '''The layer need mask. 
    The attention_dim can be any value. It's only used inside the layer.
    If set return_sequences=False, the second last dimension will be removed(average pooling). 
    e.g. Input: 32(batch)x50(sentences)x100(words)x300(vectors)
        Output: 32x50x300
        That means you input each words' vector and the layer outputs the sentence's vector.
        The dimensions of mask should be 32x50x100
        The mask is used to record how many words in each sentence.
    '''
    def __init__(self, attention_dim=128,multiple_inputs=False,return_sequences=False,**kwargs):
        self.supports_masking = True
        self.attention_dim = attention_dim
        self.multiple_inputs=multiple_inputs
        self.return_sequences=return_sequences
        super(AttentionPoolingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.multiple_inputs:
            n_in = input_shape[1][-1]
        else:
            n_in=input_shape[-1]
        print(n_in)
        #sda
        #n_in=n_in+1
        n_out = self.attention_dim
        lim = np.sqrt(6. / (n_in + n_out))
        W = K.random_uniform_variable((n_in, n_out), -lim, lim, name='{}_W'.format(self.name))
        b = K.zeros((n_out,), name='{}_b'.format(self.name))
        self.W = W
        self.b = b

        self.v = K.random_normal_variable(shape=(n_out,1),mean=0, scale=0.1, name='{}_v'.format(self.name))
        self.trainable_weights = [self.W, self.v, self.b]


    def call(self, x, mask=None):
        # x shape: 32(batch)x50(sentences)x100(words)x300(vector)
        # self.W: 300x{attention_dim}
        if self.multiple_inputs==False:
            xt=x
            xa=x
        else:
            xt=x[0]
            xa=x[1]
            mask=mask[0]
        atten = K.tanh(K.dot(xa, self.W) + self.b)  # 32x50x100x{attention_dim}
        # self.v: {attention_dim}*1
        print(xt)
        atten=K.dot(atten,self.v) # 32x50x100
        atten=K.sum(atten,axis=-1)
        atten = self.softmask(atten, mask)  # 32x50x100

        self.attention=atten
        atten=K.expand_dims(atten) # 32x50x100x1
        output = atten * xt
        if self.return_sequences==False:
            output = K.sum(output, axis=-2)  # sum the second last dimension
        return output

    def compute_output_shape(self, input_shape):
        if self.multiple_inputs:
            input_shape=input_shape[0]
        if self.return_sequences:
            return input_shape
        else:
            shape = list(input_shape)
            return tuple(shape[:-2] + shape[-1:])

    def compute_mask(self, x, mask=None):
        if self.multiple_inputs:
            mask=mask[0]
        if self.return_sequences==True:
            return mask
        elif mask is not None and K.ndim(mask)>2:
            return K.equal(K.all(K.equal(mask,False),axis=-1),False)
        else:
            return None

    def softmask(self,x, mask,axis=-1):
        '''softmax with mask, used in attention mechanism others
        '''
        y = K.exp(x)
        if mask is not None:
            mask=K.cast(mask,'float32')
            y = y * mask
        sumx = K.sum(y, axis=axis, keepdims=True) + 1e-6
        x = y / sumx
        return x
    
class ConcatLayer(Layer):
    def __init__(self,**kwargs):
        super(ConcatLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.built = True

    def call(self, inputs):
        title_xs,tag_x=inputs
        shape=K.shape(title_xs)
        if K.ndim(title_xs)==4:
            tx=K.expand_dims(tag_x,axis=-2)
            tx=K.expand_dims(tx,axis=-2)
            tx=K.tile(tx,[1,shape[1],shape[2],1])
        elif K.ndim(title_xs)==3:
            tx=K.expand_dims(tag_x,axis=-2)
            tx=K.tile(tx,[1,shape[1],1])
        outputs=concatenate([title_xs,tx],)
        return outputs

    def compute_output_shape(self, input_shape):
        return tuple(list(input_shape[0])[:-1]+[input_shape[0][-1]+input_shape[1][-1]])

    def compute_mask(self, inputs, mask=None):
        return mask[0]
    
    
class ConvDense(Layer):
    '''使用二维卷积的方法，对3维Tensor中的一维进行变换
    例如，当units=128时：
     假设输入 Tensor 32*200*300
        输出 Tensor 32*200*128
     即对最后一维进行变换
    '''
    def __init__(self, units,activation=None,**kwargs):
        super(ConvDense, self).__init__(**kwargs)
        self.filters = units
        rank=2
        strides=(1,1)
        padding='valid'
        data_format=None
        dilation_rate=(1, 1)
        use_bias=True
        kernel_initializer='glorot_uniform'
        kernel_regularizer=None
        bias_regularizer=None
        activity_regularizer=None
        kernel_constraint=None
        bias_constraint=None
        bias_initializer='zeros'
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):
        kernel_shape = (1,input_shape[-1])+(1, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        inputs=K.expand_dims(inputs)

        outputs = K.conv2d(
            inputs,
            self.kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            outputs=self.activation(outputs)

        outputs=K.sum(outputs,axis=-2)
        return outputs

    def compute_output_shape(self, input_shape):
        return tuple(list(input_shape)[:-1]+[self.filters])

    def compute_mask(self, inputs, mask=None):
        return mask

class MaskAveragePoolingLayer(Layer):
    '''AveragePooling with Mask
    The second last dimension will be removed. 
    e.g. Input: 32(batch)x50(sentences)x100(words)x300(vectors)
        Output: 32x50x300
        That means you input each words' vector and the layer outputs the sentence's vector.
        The dimensions of mask should be 32x50x100
        The mask is used to record how many words in each sentence.
    '''
    def __init__(self, **kwargs):
        self.supports_masking=True
        super(MaskAveragePoolingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        mask=K.cast(mask,'float32')
        x=x*K.expand_dims(mask)
        output = K.sum(x, axis=-2) / (K.sum(mask, axis=-1, keepdims=True) + 1e-6)
        output = K.cast(output, dtype='float32')
        return output

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        return tuple(shape[:-2] + shape[-1:])

    def compute_mask(self, x, mask=None):
        if mask is not None and K.ndim(mask)>2:
            return K.equal(K.all(K.equal(mask,False),axis=-1),False)
        else:
            return None

class ClearMaskLayer(Layer):
    def __init__(self,**kwargs):
        super(ClearMaskLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, mask=None):
        return None