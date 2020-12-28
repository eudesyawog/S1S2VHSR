import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Activation, Dropout, BatchNormalization, Concatenate, Add, Conv1D, Conv2D, Conv3D, GlobalAveragePooling1D, GlobalAveragePooling2D, MaxPooling2D, GlobalAveragePooling3D
tf.keras.backend.set_floatx('float32')

class Conv1DBlock(tf.keras.Model):
    '''
    Conv1D block with batch normalization and dropout layer
    '''
    def __init__(self,n_filters,k_size,drop,strides=1,padding_mode='valid',act='relu'):
        super(Conv1DBlock, self).__init__()
        self.conv = Conv1D (filters=n_filters, kernel_size=k_size, padding=padding_mode, strides=strides, activation=act)
        self.bn = BatchNormalization()
        self.drop_layer = Dropout(rate = drop)
    def call(self,inputs,is_training):
        conv = self.conv(inputs)
        conv = self.bn(conv)
        return self.drop_layer(conv,training=is_training)

class Conv2DBlock(tf.keras.Model):
    '''
    Conv2D block with batch normalization and dropout layer
    '''
    def __init__(self,n_filters,k_size,drop,strides=1,padding_mode='valid',act='relu'):
        super(Conv2DBlock, self).__init__()
        self.conv = Conv2D (filters=n_filters, kernel_size=k_size, padding=padding_mode, strides=strides, activation=act)
        self.bn = BatchNormalization()
        self.drop_layer = Dropout(rate = drop)
    def call(self,inputs,is_training):
        conv = self.conv(inputs)
        conv = self.bn(conv)
        return self.drop_layer(conv,training=is_training)

class Conv2DAndMaxPoolingBlock(tf.keras.Model):
    '''
    Conv2D block with batch normalization, maxpooling and dropout layer
    '''
    def __init__(self,n_filters,k_size,drop,pool_size,strides_conv=1,strides_pool=2,padding_mode='valid',act='relu'):
        super(Conv2DAndMaxPoolingBlock, self).__init__()
        self.conv = Conv2D (filters=n_filters, kernel_size=k_size, padding=padding_mode, strides=strides_conv, activation=act)
        self.bn = BatchNormalization()
        self.max_pool = MaxPooling2D(pool_size=pool_size, strides=strides_pool)
        self.drop_layer = Dropout(rate = drop)
    def call(self,inputs,is_training):
        conv = self.conv(inputs)
        conv = self.bn(conv)
        conv = self.max_pool(conv)
        return self.drop_layer(conv,training=is_training)

class Conv3DBlock(tf.keras.Model):
    '''
    Conv3D block with batch normalization and dropout layer
    '''
    def __init__(self,n_filters,k_size,drop,strides=1,padding_mode='valid',act='relu'):
        super(Conv3DBlock, self).__init__()
        self.conv = Conv3D (filters=n_filters, kernel_size=k_size, padding=padding_mode, strides=strides, activation=act)
        self.bn = BatchNormalization()
        self.drop_layer = Dropout(rate = drop)
    def call(self,inputs,is_training):
        conv = self.conv(inputs)
        conv = self.bn(conv)
        return self.drop_layer(conv,training=is_training)

class FC(tf.keras.Model):
    '''
    Dense layer with batch normalization 
    '''
    def __init__(self,num_units,act='relu'):
        super(FC,self).__init__()
        self.dense = Dense(num_units, activation=act)
        self.bn = BatchNormalization()
    def call(self,inputs):
        return self.bn ( self.dense(inputs) )

class SoftMax(Layer):
    '''
    Output layer with predictions for n_classes 
    '''
    def __init__(self,n_classes):
        super(SoftMax,self).__init__()
        self.dense = Dense(n_classes,activation='softmax')
    def call(self,inputs):
        return self.dense(inputs)

class CNN1D_Encoder(tf.keras.Model):
    '''
    1D-CNN encoder
    '''
    def __init__(self,n_filters,drop):
        super(CNN1D_Encoder,self).__init__(name='CNN1D_Encoder')
        self.block1 = Conv1DBlock(n_filters,5,drop)
        self.block2 = Conv1DBlock(n_filters,3,drop,strides=2)
        self.block3 = Conv1DBlock(n_filters*2,3,drop)
        self.block4 = Conv1DBlock(n_filters*2,1,drop)
        self.gap = GlobalAveragePooling1D()
    def call(self,inputs,is_training):
        b1 = self.block1(inputs,is_training)
        b2 = self.block2(b1,is_training)
        b3 = self.block3(b2,is_training)
        b4 = self.block4(b3,is_training)
        return self.gap(b4)

class CNN2D_Encoder(tf.keras.Model):
    '''
    2D-CNN encoder
    '''
    def __init__(self,n_filters,drop):
        super(CNN2D_Encoder,self).__init__(name='CNN2D_Encoder')
        self.block1 = Conv2DBlock(n_filters,3,drop)
        self.block2 = Conv2DBlock(n_filters,3,drop)
        self.block3 = Conv2DBlock(n_filters*2,3,drop)
        self.block4 = Conv2DBlock(n_filters*2,1,drop)
        self.gap = GlobalAveragePooling2D()
    def call(self,inputs,is_training):
        b1 = self.block1(inputs,is_training)
        b2 = self.block2(b1,is_training)
        b3 = self.block3(b2,is_training)
        b4 = self.block4(b3,is_training)
        return self.gap(b4)

class CNN3D_Encoder(tf.keras.Model):
    '''
    3D-CNN encoder
    '''
    def __init__(self,n_filters,drop):
        super(CNN3D_Encoder,self).__init__(name='CNN3D_Encoder')
        self.block1 = Conv3DBlock(n_filters,3,drop)
        self.block2 = Conv3DBlock(n_filters,3,drop,(1,1,2))
        self.block3 = Conv3DBlock(n_filters*2,3,drop,(1,1,2))
        self.block4 = Conv3DBlock(n_filters*2,1,drop)
        self.gap = GlobalAveragePooling3D()
    def call(self,inputs,is_training):
        b1 = self.block1(inputs,is_training)
        b2 = self.block2(b1,is_training)
        b3 = self.block3(b2,is_training)
        b4 = self.block4(b3,is_training)
        return self.gap(b4)

class Spot_Branch(tf.keras.Model):
    '''
    SPOT CNN encoder
    '''
    def __init__(self,n_filters,drop):
        super(Spot_Branch,self).__init__(name='Spot_Branch')
        self.block1 = Conv2DAndMaxPoolingBlock(n_filters,7,drop,3)
        self.block2 = Conv2DBlock(n_filters*2,5,drop)
        self.block3 = Conv2DAndMaxPoolingBlock(n_filters*2,3,drop,3,padding_mode='same')
        self.block4 = Conv2DBlock(n_filters*2,3,drop)
        self.block5 = Conv2DBlock(n_filters*2,1,drop)
        self.concat = Concatenate()
        self.gap = GlobalAveragePooling2D()
    def call(self,inputs_ms,inputs_pan,is_training):
        b1 = self.block1(inputs_pan,is_training)
        b2 = self.block2(b1,is_training)
        concat = self.concat([b2,inputs_ms])
        b3 = self.block3(concat,is_training)
        b4 = self.block4(b3,is_training)
        b5 = self.block5(b4,is_training)
        return self.gap(b5)

class Model_S1S2SPOT(tf.keras.Model):
    '''
    Model for Sentinel-1, Sentinel-2 and SPOT fusion
    '''
    def __init__(self,drop,n_classes,n_filters,fusion,sensor,num_units=512):
        super(Model_S1S2SPOT, self).__init__(name='Model_S1S2SPOT')
        if 's1-1D' in sensor:
            self.s1_branch = CNN1D_Encoder(n_filters,drop)
        elif 's1-2D'in sensor:
            self.s1_branch = CNN2D_Encoder(n_filters,drop)
        elif 's1-3D'in sensor:
            self.s1_branch = CNN3D_Encoder(n_filters,drop)
        if 's2-1D' in sensor:
            self.s2_branch = CNN1D_Encoder(n_filters,drop)
        elif 's2-2D' in sensor:
            self.s2_branch = CNN2D_Encoder(n_filters,drop)
        elif 's2-3D' in sensor:
            self.s2_branch = CNN3D_Encoder(n_filters,drop)
        self.spot_branch = Spot_Branch(n_filters,drop)
        if fusion == 'concat':
            self.fusion = Concatenate()
        elif fusion == 'add':
            self.fusion = Add()
        self.dense1 = FC(num_units)
        self.dense2 = FC(num_units)
        self.softmax1 = SoftMax(n_classes)
        self.softmax2 = SoftMax(n_classes)
        self.softmax3 = SoftMax(n_classes)
        self.softmax4 = SoftMax(n_classes)
    def call(self,x_s1, x_s2, x_ms, x_pan, is_training):
        feat_s1 = self.s1_branch(x_s1,is_training)
        feat_s2 = self.s2_branch(x_s2,is_training)
        feat_spot = self.spot_branch(x_ms,x_pan,is_training)
        feat_fused = self.fusion([feat_s1,feat_s2,feat_spot])
        fused_pred = self.softmax1( self.dense2( self.dense1(feat_fused) ) )
        s1_pred = self.softmax2(feat_s1)
        s2_pred = self.softmax3(feat_s2)
        spot_pred = self.softmax4(feat_spot)
        return s1_pred,s2_pred,spot_pred,fused_pred
    def getEmbedding(self, x_s1, x_s2, x_ms, x_pan, is_training=False):
        feat_spot = self.spot_branch(x_ms,x_pan,is_training)
        feat_s1 = self.s1_branch(x_s1,is_training)
        feat_s2 = self.s2_branch(x_s2,is_training)
        feat_fused = self.fusion([feat_s1,feat_s2,feat_spot])
        embedding = self.dense2( self.dense1(feat_fused) )
        return embedding

class Model_S1S2(tf.keras.Model):
    '''
    Model for Sentinel-1 and Sentinel-2 fusion
    '''
    def __init__(self,drop,n_classes,n_filters,fusion,sensor,num_units=512):
        super(Model_S1S2, self).__init__(name='Model_S1S2')
        if 's1-1D' in sensor:
            self.s1_branch = CNN1D_Encoder(n_filters,drop)
        elif 's1-2D'in sensor:
            self.s1_branch = CNN2D_Encoder(n_filters,drop)
        elif 's1-3D'in sensor:
            self.s1_branch = CNN3D_Encoder(n_filters,drop)
        if 's2-1D' in sensor:
            self.s2_branch = CNN1D_Encoder(n_filters,drop)
        elif 's2-2D' in sensor:
            self.s2_branch = CNN2D_Encoder(n_filters,drop)
        elif 's2-3D' in sensor:
            self.s2_branch = CNN3D_Encoder(n_filters,drop)
        if fusion == 'concat':
            self.fusion = Concatenate()
        elif fusion == 'add':
            self.fusion = Add()
        self.dense1 = FC(num_units)
        self.dense2 = FC(num_units)
        self.softmax1 = SoftMax(n_classes)
        self.softmax2 = SoftMax(n_classes)
        self.softmax3 = SoftMax(n_classes)
    def call(self,x_s1, x_s2, is_training):
        feat_s1 = self.s1_branch(x_s1,is_training)
        feat_s2 = self.s2_branch(x_s2,is_training)
        feat_fused = self.fusion([feat_s1,feat_s2])
        fused_pred = self.softmax1( self.dense2( self.dense1(feat_fused) ) )
        s1_pred = self.softmax2(feat_s1)
        s2_pred = self.softmax3(feat_s2)
        return s1_pred,s2_pred,fused_pred
    def getEmbedding(self, x_s1, x_s2, is_training=False):
        feat_s1 = self.s1_branch(x_s1,is_training)
        feat_s2 = self.s2_branch(x_s2,is_training)
        feat_fused = self.fusion([feat_s1,feat_s2])
        embedding = self.dense2( self.dense1(feat_fused) )
        return embedding

class Model_S2SPOT(tf.keras.Model):
    '''
    Model for Sentinel-2 and SPOT fusion
    '''
    def __init__(self,drop,n_classes,n_filters,fusion,sensor,num_units=512):
        super(Model_S2SPOT, self).__init__(name='Model_S2SPOT')
        if 's2-1D' in sensor:
            self.s2_branch = CNN1D_Encoder(n_filters,drop)
        elif 's2-2D' in sensor:
            self.s2_branch = CNN2D_Encoder(n_filters,drop)
        elif 's2-3D' in sensor:
            self.s2_branch = CNN3D_Encoder(n_filters,drop)
        self.spot_branch = Spot_Branch(n_filters,drop)
        if fusion == 'concat':
            self.fusion = Concatenate()
        elif fusion == 'add':
            self.fusion = Add()
        self.dense1 = FC(num_units)
        self.dense2 = FC(num_units)
        self.softmax1 = SoftMax(n_classes)
        self.softmax2 = SoftMax(n_classes)
        self.softmax3 = SoftMax(n_classes)
    def call(self, x_s2, x_ms, x_pan, is_training):
        feat_s2 = self.s2_branch(x_s2,is_training)
        feat_spot = self.spot_branch(x_ms,x_pan,is_training)
        feat_fused = self.fusion([feat_s2,feat_spot])
        fused_pred = self.softmax1( self.dense2( self.dense1(feat_fused) ) )
        s2_pred = self.softmax2(feat_s2)
        spot_pred = self.softmax3(feat_spot)
        return s2_pred,spot_pred,fused_pred
    def getEmbedding(self, x_s2, x_ms, x_pan, is_training=False):
        feat_s2 = self.s2_branch(x_s2,is_training)
        feat_spot = self.spot_branch(x_ms,x_pan,is_training)
        feat_fused = self.fusion([feat_s2,feat_spot])
        embedding = self.dense2( self.dense1(feat_fused) )
        return embedding

class Model_S1(tf.keras.Model):
    '''
    Model for Sentinel-1 alone
    '''
    def __init__(self,drop,n_classes,n_filters,sensor,num_units=512):
        super(Model_S1, self).__init__(name='Model_S1')
        if 's1-1D' in sensor:
            self.s1_branch = CNN1D_Encoder(n_filters,drop)
        elif 's1-2D'in sensor:
            self.s1_branch = CNN2D_Encoder(n_filters,drop)
        elif 's1-3D'in sensor:
            self.s1_branch = CNN3D_Encoder(n_filters,drop)
        self.dense1 = FC(num_units)
        self.dense2 = FC(num_units)
        self.softmax = SoftMax(n_classes)
    def call(self,x_s1, is_training):
        feat = self.s1_branch(x_s1,is_training)
        pred = self.softmax( self.dense2( self.dense1(feat) ) )
        return pred
    def getEmbedding(self, x_s1, is_training=False):
        feat = self.s1_branch(x_s1,is_training)
        embedding = self.dense2( self.dense1(feat) )
        return embedding

class Model_S2(tf.keras.Model):
    '''
    Model for Sentinel-2 alone
    '''
    def __init__(self,drop,n_classes,n_filters,sensor,num_units=512):
        super(Model_S2, self).__init__(name='Model_S2')
        if 's2-1D' in sensor:
            self.s2_branch = CNN1D_Encoder(n_filters,drop)
        elif 's2-2D' in sensor:
            self.s2_branch = CNN2D_Encoder(n_filters,drop)
        elif 's2-3D' in sensor:
            self.s2_branch = CNN3D_Encoder(n_filters,drop)
        self.dense1 = FC(num_units)
        self.dense2 = FC(num_units)
        self.softmax = SoftMax(n_classes)
    def call(self,x_s2, is_training):
        feat = self.s2_branch(x_s2,is_training)
        pred = self.softmax( self.dense2( self.dense1(feat) ) )
        return pred
    def getEmbedding(self, x_s2, is_training=False):
        feat = self.s2_branch(x_s2,is_training)
        embedding = self.dense2( self.dense1(feat) )
        return embedding

class Model_SPOT(tf.keras.Model):
    '''
    Model for SPOT alone
    '''
    def __init__(self,drop,n_classes,n_filters,num_units=512):
        super(Model_SPOT, self).__init__(name='Model_SPOT')
        self.spot_branch = Spot_Branch(n_filters,drop)
        self.dense1 = FC(num_units)
        self.dense2 = FC(num_units)
        self.softmax = SoftMax(n_classes)
    def call(self,x_ms, x_pan, is_training):
        feat = self.spot_branch(x_ms,x_pan,is_training)
        pred = self.softmax( self.dense2( self.dense1(feat) ) )
        return pred
    def getEmbedding(self, x_ms, x_pan, is_training=False):
        feat = self.spot_branch(x_ms,x_pan,is_training)
        embedding = self.dense2( self.dense1(feat) )
        return embedding