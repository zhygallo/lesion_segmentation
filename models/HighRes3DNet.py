from keras.models import Model
from keras.layers import Input, Conv3D, Softmax, Add, BatchNormalization, PReLU, Concatenate


def get_model(input_shape, n_channels=1, weights_path=None):
    input_shape = (input_shape[0], input_shape[1], input_shape[2], n_channels)

    input_layer = Input(shape=input_shape, name='input_1')

    conv1 = Conv3D(16, (3, 3, 3), padding='same', data_format='channels_last')(input_layer)
    batch_norm1 = BatchNormalization()(conv1)
    prelu1 = PReLU()(batch_norm1)

    batch_norm2 = BatchNormalization()(prelu1)
    prelu2 = PReLU()(batch_norm2)
    conv2 = Conv3D(16, (3, 3, 3), padding='same', dilation_rate=1)(prelu2)
    batch_norm3 = BatchNormalization()(conv2)
    prelu3 = PReLU()(batch_norm3)
    conv3 = Conv3D(16, (3, 3, 3), padding='same', dilation_rate=1)(prelu3)

    add1 = Add()([conv3, prelu1])

    batch_norm4 = BatchNormalization()(add1)
    prelu4 = PReLU()(batch_norm4)
    conv4 = Conv3D(16, (3, 3, 3), padding='same', dilation_rate=1)(prelu4)
    batch_norm5 = BatchNormalization()(conv4)
    prelu5 = PReLU()(batch_norm5)
    conv5 = Conv3D(16, (3, 3, 3), padding='same', dilation_rate=1)(prelu5)

    add2 = Add()([conv5, add1])

    batch_norm6 = BatchNormalization()(add2)
    prelu6 = PReLU()(batch_norm6)
    conv6 = Conv3D(16, (3, 3, 3), padding='same', dilation_rate=1)(prelu6)
    batch_norm7 = BatchNormalization()(conv6)
    prelu7 = PReLU()(batch_norm7)
    conv7 = Conv3D(16, (3, 3, 3), padding='same', dilation_rate=1)(prelu7)

    add3 = Add()([conv7, add2])

    batch_norm8 = BatchNormalization()(add3)
    prelu8 = PReLU()(batch_norm8)
    conv8 = Conv3D(32, (3, 3, 3), padding='same', dilation_rate=2)(prelu8)
    batch_norm9 = BatchNormalization()(conv8)
    prelu9 = PReLU()(batch_norm9)
    conv9 = Conv3D(32, (3, 3, 3), padding='same', dilation_rate=2)(prelu9)

    concat1 = Concatenate()([add3] * 2)
    add4 = Add()([conv9, concat1])

    batch_norm10 = BatchNormalization()(add4)
    prelu10 = PReLU()(batch_norm10)
    conv10 = Conv3D(32, (3, 3, 3), padding='same', dilation_rate=2)(prelu10)
    batch_norm11 = BatchNormalization()(conv10)
    prelu11 = PReLU()(batch_norm11)
    conv11 = Conv3D(32, (3, 3, 3), padding='same', dilation_rate=2)(prelu11)

    add5 = Add()([conv11, add4])

    batch_norm12 = BatchNormalization()(add5)
    prelu12 = PReLU()(batch_norm12)
    conv12 = Conv3D(32, (3, 3, 3), padding='same', dilation_rate=2)(prelu12)
    batch_norm13 = BatchNormalization()(conv12)
    prelu13 = PReLU()(batch_norm13)
    conv13 = Conv3D(32, (3, 3, 3), padding='same', dilation_rate=2)(prelu13)

    add6 = Add()([conv13, add5])

    batch_norm14 = BatchNormalization()(add6)
    prelu14 = PReLU()(batch_norm14)
    conv14 = Conv3D(64, (3, 3, 3), padding='same', dilation_rate=4)(prelu14)
    batch_norm15 = BatchNormalization()(conv14)
    prelu15 = PReLU()(batch_norm15)
    conv15 = Conv3D(64, (3, 3, 3), padding='same', dilation_rate=4)(prelu15)

    concat2 = Concatenate()([add6] * 2)
    add7 = Add()([conv15, concat2])

    batch_norm16 = BatchNormalization()(add7)
    prelu16 = PReLU()(batch_norm16)
    conv16 = Conv3D(64, (3, 3, 3), padding='same', dilation_rate=4)(prelu16)
    batch_norm17 = BatchNormalization()(conv16)
    prelu17 = PReLU()(batch_norm17)
    conv17 = Conv3D(64, (3, 3, 3), padding='same', dilation_rate=4)(prelu17)

    add8 = Add()([conv17, add7])

    batch_norm18 = BatchNormalization()(add8)
    prelu18 = PReLU()(batch_norm18)
    conv18 = Conv3D(64, (3, 3, 3), padding='same', dilation_rate=4)(prelu18)
    batch_norm19 = BatchNormalization()(conv18)
    prelu19 = PReLU()(batch_norm19)
    conv19 = Conv3D(64, (3, 3, 3), padding='same', dilation_rate=4)(prelu19)

    add9 = Add()([conv19, add8])

    conv20 = Conv3D(160, (1, 1, 1))(add9)

    conv21 = Conv3D(2, (1, 1, 1))(conv20)
    softmax = Softmax()(conv21)

    model = Model(inputs=[input_layer], outputs=[softmax])

    return model
