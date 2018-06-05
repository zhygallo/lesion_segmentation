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

    add1 = Add()([conv2, prelu1])

    batch_norm3 = BatchNormalization()(add1)
    prelu3 = PReLU()(batch_norm3)
    conv3 = Conv3D(16, (3, 3, 3), padding='same', dilation_rate=1)(prelu3)

    add2 = Add()([conv3, add1])

    batch_norm4 = BatchNormalization()(add2)
    prelu4 = PReLU()(batch_norm4)
    conv4 = Conv3D(32, (3, 3, 3), padding='same', dilation_rate=2)(prelu4)

    concat1 = Concatenate()([add2] * 2)
    add4 = Add()([conv4, concat1])

    batch_norm5 = BatchNormalization()(add4)
    prelu5 = PReLU()(batch_norm5)
    conv5 = Conv3D(32, (3, 3, 3), padding='same', dilation_rate=2)(prelu5)

    add5 = Add()([conv5, add4])

    batch_norm6 = BatchNormalization()(add5)
    prelu6 = PReLU()(batch_norm6)
    conv6 = Conv3D(64, (3, 3, 3), padding='same', dilation_rate=4)(prelu6)

    concat2 = Concatenate()([add5] * 2)
    add7 = Add()([conv6, concat2])

    batch_norm7 = BatchNormalization()(add7)
    prelu7 = PReLU()(batch_norm7)
    conv7 = Conv3D(64, (3, 3, 3), padding='same', dilation_rate=4)(prelu7)

    add8 = Add()([conv7, add7])

    conv8 = Conv3D(40, (1, 1, 1))(add8)

    conv9 = Conv3D(2, (1, 1, 1), activation='softmax')(conv8)

    model = Model(inputs=[input_layer], outputs=[conv9])

    return model
