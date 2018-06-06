from keras.models import Model
from keras.layers import Input, Conv3D, Add, BatchNormalization, PReLU, Concatenate


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
    conv8 = Conv3D(16, (3, 3, 3), padding='same', dilation_rate=1)(prelu8)
    batch_norm9 = BatchNormalization()(conv8)
    prelu9 = PReLU()(batch_norm9)
    conv9 = Conv3D(16, (3, 3, 3), padding='same', dilation_rate=1)(prelu9)

    add4 = Add()([conv9, add3])

    batch_norm10 = BatchNormalization()(add4)
    prelu10 = PReLU()(batch_norm10)
    conv10 = Conv3D(32, (3, 3, 3), padding='same', dilation_rate=2)(prelu10)
    batch_norm11 = BatchNormalization()(conv10)
    prelu11 = PReLU()(batch_norm11)
    conv11 = Conv3D(32, (3, 3, 3), padding='same', dilation_rate=2)(prelu11)

    concat1 = Concatenate()([add4] * 2)
    add5 = Add()([conv11, concat1])

    batch_norm12 = BatchNormalization()(add5)
    prelu12 = PReLU()(batch_norm12)
    conv12 = Conv3D(32, (3, 3, 3), padding='same', dilation_rate=2)(prelu12)
    batch_norm13 = BatchNormalization()(conv12)
    prelu13 = PReLU()(batch_norm13)
    conv13 = Conv3D(32, (3, 3, 3), padding='same', dilation_rate=2)(prelu13)

    add6 = Add()([conv13, add5])

    batch_norm14 = BatchNormalization()(add6)
    prelu14 = PReLU()(batch_norm14)
    conv14 = Conv3D(32, (3, 3, 3), padding='same', dilation_rate=2)(prelu14)
    batch_norm15 = BatchNormalization()(conv14)
    prelu15 = PReLU()(batch_norm15)
    conv15 = Conv3D(32, (3, 3, 3), padding='same', dilation_rate=2)(prelu15)

    add7 = Add()([conv15, add6])

    batch_norm16 = BatchNormalization()(add7)
    prelu16 = PReLU()(batch_norm16)
    conv16 = Conv3D(32, (3, 3, 3), padding='same', dilation_rate=2)(prelu16)
    batch_norm17 = BatchNormalization()(conv16)
    prelu17 = PReLU()(batch_norm17)
    conv17 = Conv3D(32, (3, 3, 3), padding='same', dilation_rate=2)(prelu17)

    add8 = Add()([conv17, add7])

    batch_norm18 = BatchNormalization()(add8)
    prelu18 = PReLU()(batch_norm18)
    conv18 = Conv3D(64, (3, 3, 3), padding='same', dilation_rate=4)(prelu18)
    batch_norm19 = BatchNormalization()(conv18)
    prelu19 = PReLU()(batch_norm19)
    conv19 = Conv3D(64, (3, 3, 3), padding='same', dilation_rate=4)(prelu19)

    concat2 = Concatenate()([add8] * 2)
    add9 = Add()([conv19, concat2])

    batch_norm20 = BatchNormalization()(add9)
    prelu20 = PReLU()(batch_norm20)
    conv20 = Conv3D(64, (3, 3, 3), padding='same', dilation_rate=4)(prelu20)
    batch_norm21 = BatchNormalization()(conv20)
    prelu21 = PReLU()(batch_norm21)
    conv21 = Conv3D(64, (3, 3, 3), padding='same', dilation_rate=4)(prelu21)

    add10 = Add()([conv21, add9])

    batch_norm22 = BatchNormalization()(add10)
    prelu22 = PReLU()(batch_norm22)
    conv22 = Conv3D(64, (3, 3, 3), padding='same', dilation_rate=4)(prelu22)
    batch_norm23 = BatchNormalization()(conv22)
    prelu23 = PReLU()(batch_norm23)
    conv23 = Conv3D(64, (3, 3, 3), padding='same', dilation_rate=4)(prelu23)

    add11 = Add()([conv23, add10])

    batch_norm24 = BatchNormalization()(add11)
    prelu24 = PReLU()(batch_norm24)
    conv24 = Conv3D(64, (3, 3, 3), padding='same', dilation_rate=4)(prelu24)
    batch_norm25 = BatchNormalization()(conv24)
    prelu25 = PReLU()(batch_norm25)
    conv25 = Conv3D(64, (3, 3, 3), padding='same', dilation_rate=4)(prelu25)

    add12 = Add()([conv25, add11])

    conv26 = Conv3D(160, (1, 1, 1))(add12)

    conv27 = Conv3D(2, (1, 1, 1), activation='softmax')(conv26)

    model = Model(inputs=[input_layer], outputs=[conv27])

    return model
