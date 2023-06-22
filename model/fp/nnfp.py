# -*- coding: utf-8 -*-
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
""" nnfp.py

'Neural Audio Fingerprint for High-specific Audio Retrieval based on 
Contrastive Learning', https://arxiv.org/abs/2010.11910

USAGE:
    
    Please see test() in the below.
    
"""
import numpy as np
import tensorflow as tf
assert tf.__version__ >= "2.0"


class ConvLayer(tf.keras.layers.Layer):
    """
    Separable convolution layer
    
    Arguments
    ---------
    hidden_ch: (int)
    strides: [(int, int), (int, int)]
    norm: 'layer_norm1d' for normalization on Freq axis. (default)
          'layer_norm2d' for normalization on on FxT space 
          'batch_norm' or else, batch-normalization
    
    Input
    -----
    x: (B,F,T,1)
    
    [Conv1x3]>>[ELU]>>[BN]>>[Conv3x1]>>[ELU]>>[BN]
    
    Output
    ------
    x: (B,F,T,C) with {F=F/stride, T=T/stride, C=hidden_ch}
    
    """

    def __init__(self,
                 hidden_ch=128,
                 strides=[(1,1),(1,1)],
                 norm='layer_norm2d'):
        super(ConvLayer, self).__init__()

        self.conv2d_1x3 = tf.keras.layers.Conv2D(hidden_ch,
                                                 kernel_size=(1, 3),
                                                 strides=strides[0],
                                                 padding='SAME',
                                                 dilation_rate=(1, 1),
                                                 kernel_initializer='glorot_uniform',
                                                 bias_initializer='zeros')
        self.conv2d_3x1 = tf.keras.layers.Conv2D(hidden_ch,
                                                 kernel_size=(3, 1),
                                                 strides=strides[1],
                                                 padding='SAME',
                                                 dilation_rate=(1, 1),
                                                 kernel_initializer='glorot_uniform',
                                                 bias_initializer='zeros')
        if norm == 'layer_norm1d':
            self.BN_1x3 = tf.keras.layers.LayerNormalization(axis=-1)
            self.BN_3x1 = tf.keras.layers.LayerNormalization(axis=-1)
        elif norm == 'layer_norm2d':
            self.BN_1x3 = tf.keras.layers.LayerNormalization(axis=(1, 2, 3))
            self.BN_3x1 = tf.keras.layers.LayerNormalization(axis=(1, 2, 3))
        else:
            self.BN_1x3 = tf.keras.layers.BatchNormalization(axis=-1) # Fix axis: 2020 Apr20
            self.BN_3x1 = tf.keras.layers.BatchNormalization(axis=-1)
            
        self.forward = tf.keras.Sequential([self.conv2d_1x3,
                                            tf.keras.layers.ELU(),
                                            self.BN_1x3,
                                            self.conv2d_3x1,
                                            tf.keras.layers.ELU(),
                                            self.BN_3x1
                                            ])

       
    def call(self, x):
        return self.forward(x)


class DivEncLayer(tf.keras.layers.Layer):
    """
    Multi-head projection a.k.a. 'divide and encode' layer:
        
    • The concept of 'divide and encode' was discovered  in Lai et.al.,
     'Simultaneous Feature Learning and Hash Coding with Deep Neural Networks',
      2015. https://arxiv.org/abs/1504.03410
    • It was also adopted in Gfeller et.al. 'Now Playing: Continuo-
      us low-power music recognition', 2017. https://arxiv.org/abs/1711.10958
    
    Arguments
    ---------
    q: (int) number of slices as 'slice_length = input_dim / q'
    unit_dim: [(int), (int)]
    norm: 'layer_norm1d' or 'layer_norm2d' uses 1D-layer normalization on the feature.
          'batch_norm' or else uses batch normalization. Default is 'layer_norm2d'.

    Input
    -----
    x: (B,1,1,C)
    
    Returns
    -------
    emb: (B,Q)
    
    """
    def __init__(self, q=128, unit_dim=[32, 1], norm='batch_norm'):
        super(DivEncLayer, self).__init__()

        self.q = q
        self.unit_dim = unit_dim
        self.norm = norm
        
        if norm in ['layer_norm1d', 'layer_norm2d']:
            self.BN = [tf.keras.layers.LayerNormalization(axis=-1) for i in range(q)]
        else:
            self.BN = [tf.keras.layers.BatchNormalization(axis=-1) for i in range(q)]
            
        self.split_fc_layers = self._construct_layers() 


    def build(self, input_shape):
        # Prepare output embedding variable for dynamic batch-size 
        self.slice_length = int(input_shape[-1] / self.q)

 
    def _construct_layers(self):
        layers = list()
        for i in range(self.q): # q: num_slices
            layers.append(tf.keras.Sequential([tf.keras.layers.Dense(self.unit_dim[0], activation='elu'),
                                               #self.BN[i],
                                               tf.keras.layers.Dense(self.unit_dim[1])]))
        return layers


    @tf.function
    def _split_encoding(self, x_slices):
        """
        Input: (B,Q,S)
        Returns: (B,Q)
        
        """
        out = list()
        for i in range(self.q):
            out.append(self.split_fc_layers[i](x_slices[:, i, :]))
        return tf.concat(out, axis=1)

    
    def call(self, x): # x: (B,1,1,2048)
        x = tf.reshape(x, shape=[x.shape[0], self.q, -1]) # (B,Q,S); Q=num_slices; S=slice length; (B,128,8 or 16)
        return self._split_encoding(x)


class FingerPrinter(tf.keras.Model):
    """
    Fingerprinter: 'Neural Audio Fingerprint for High-specific Audio Retrieval
        based on Contrastive Learning', https://arxiv.org/abs/2010.11910
    
    IN >> [Convlayer]x8 >> [DivEncLayer] >> [L2Normalizer] >> OUT 
    
    Arguments
    ---------
    input_shape: tuple (int), not including the batch size
    front_hidden_ch: (list)
    front_strides: (list)
    emb_sz: (int) default=128
    fc_unit_dim: (list) default=[32,1]
    norm: 'layer_norm1d' for normalization on Freq axis. 
          'layer_norm2d' for normalization on on FxT space (default).
          'batch_norm' or else, batch-normalization.
    use_L2layer: True (default)
    
    • Note: batch-normalization will not work properly with TPUs.
                    
    
    Input
    -----
    x: (B,F,T,1)
    
        
    Returns
    -------
    emb: (B,Q) 
    
    """
    def __init__(self,
                 front_hidden_ch=[128, 128, 256, 256, 512, 512, 1024, 1024],
                 front_strides=[[(1,2), (2,1)], [(1,2), (2,1)],
                                [(1,2), (2,1)], [(1,2), (2,1)],
                                [(1,1), (2,1)], [(1,2), (2,1)],
                                [(1,1), (2,1)], [(1,2), (2,1)]],
                 emb_sz=128, # q
                 fc_unit_dim=[32,1],
                 norm='layer_norm2d',
                 use_L2layer=True):
        super(FingerPrinter, self).__init__()

        assert len(front_hidden_ch) == len(front_strides), \
            "front_hidden_ch and front_strides must have the same length."

        self.front_hidden_ch = front_hidden_ch
        self.front_strides = front_strides
        self.emb_sz = emb_sz
        self.norm = norm
        self.use_L2layer = use_L2layer

        # Front (sep-)conv layers
        self.n_clayers = len(front_strides)
        self.front_conv = tf.keras.Sequential(name='ConvLayers')
        if ((front_hidden_ch[-1] % emb_sz) != 0):
            front_hidden_ch[-1] = ((front_hidden_ch[-1]//emb_sz) + 1) * emb_sz
        for i in range(self.n_clayers):
            self.front_conv.add(ConvLayer(hidden_ch=front_hidden_ch[i],
                                          strides=front_strides[i], 
                                          norm=norm))
        self.front_conv.add(tf.keras.layers.Flatten()) # (B,F',T',C) >> (B,D)

        # Divide & Encoder layer
        self.div_enc = DivEncLayer(q=emb_sz, unit_dim=fc_unit_dim, norm=norm)

    @tf.function
    def call(self, inputs):
        """ Input: (B,F,T,1)"""

        x = self.front_conv(inputs) # (B,D) with D = (T/2^4) x last_hidden_ch
        x = self.div_enc(x) # (B,Q)
        if self.use_L2layer:
            return tf.math.l2_normalize(x, axis=1) 
        else:
            return x


def get_fingerprinter(cfg, trainable=False):
    """
    Input length : 1s or 2s
    
    Arguements
    ----------
    cfg : (dict)
        created from the '.yaml' located in /config dicrectory

    Returns
    -------
    <tf.keras.Model> FingerPrinter object
    
    """
    emb_sz = cfg['MODEL']['EMB_SZ']
    norm = cfg['MODEL']['BN']
    fc_unit_dim = [32, 1]
    
    m = FingerPrinter(emb_sz=emb_sz,
                      fc_unit_dim=fc_unit_dim,
                      norm=norm)
    m.trainable = trainable
    return m
    

def test():
    input_1s = tf.constant(np.random.randn(3,256,32,1), dtype=tf.float32) # BxFxTx1
    fprinter = FingerPrinter(emb_sz=128, fc_unit_dim=[32, 1], norm='layer_norm2d')
    emb_1s = fprinter(input_1s) # BxD
    
    input_2s = tf.constant(np.random.randn(3,256,63,1), dtype=tf.float32) # BxFxTx1
    fprinter = FingerPrinter(emb_sz=128, fc_unit_dim=[32, 1], norm='layer_norm2d')
    emb_2s = fprinter(input_2s)
    #%timeit -n 10 fprinter(_input) # 27.9ms
"""
Total params: 19,224,576
Trainable params: 19,224,576
Non-trainable params: 0

"""
