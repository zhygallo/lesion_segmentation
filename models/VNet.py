from keras.models import Model
from keras.layers import Input, Conv3D, Conv3DTranspose, Concatenate, Add, PReLU, Softmax


def get_model(input_shape, n_channels=1, weights_path=None):
    input_shape = (input_shape[0], input_shape[1], input_shape[2], n_channels)

    input_layer = Input(shape=input_shape, name='input_1')
    conv1_1 = Conv3D(16, (5, 5, 5), padding='same', data_format='channels_last')(input_layer)
    prelu1_1 = PReLU()(conv1_1)
    concat1 = Concatenate()([input_layer] * 16)
    add1 = Add()([prelu1_1, concat1])
    downconv1 = Conv3D(32, (2, 2, 2), strides=(2, 2, 2))(add1)
    prelu1_2 = PReLU()(downconv1)

    conv2_1 = Conv3D(32, (5, 5, 5), padding='same')(prelu1_2)
    prelu2_1 = PReLU()(conv2_1)
    conv2_2 = Conv3D(32, (5, 5, 5), padding='same')(prelu2_1)
    prelu2_2 = PReLU()(conv2_2)
    add2 = Add()([prelu2_2, prelu1_2])
    downconv2 = Conv3D(64, (2, 2, 2), strides=(2, 2, 2))(add2)
    prelu2_3 = PReLU()(downconv2)

    conv3_1 = Conv3D(64, (5, 5, 5), padding='same')(prelu2_3)
    prelu3_1 = PReLU()(conv3_1)
    conv3_2 = Conv3D(64, (5, 5, 5), padding='same')(prelu3_1)
    prelu3_2 = PReLU()(conv3_2)
    conv3_3 = Conv3D(64, (5, 5, 5), padding='same')(prelu3_2)
    prelu3_3 = PReLU()(conv3_3)
    add3 = Add()([prelu3_3, prelu2_3])
    downconv3 = Conv3D(128, (2, 2, 2), strides=(2, 2, 2))(add3)
    prelu3_4 = PReLU()(downconv3)

    conv4_1 = Conv3D(128, (5, 5, 5), padding='same')(prelu3_4)
    prelu4_1 = PReLU()(conv4_1)
    conv4_2 = Conv3D(128, (5, 5, 5), padding='same')(prelu4_1)
    prelu4_2 = PReLU()(conv4_2)
    conv4_3 = Conv3D(128, (5, 5, 5), padding='same')(prelu4_2)
    prelu4_3 = PReLU()(conv4_3)
    add4 = Add()([prelu4_3, prelu3_4])
    downconv4 = Conv3D(256, (2, 2, 2), strides=(2, 2, 2))(add4)
    prelu4_4 = PReLU()(downconv4)

    conv5_1 = Conv3D(256, (5, 5, 4), padding='same')(prelu4_4)
    prelu5_1 = PReLU()(conv5_1)
    conv5_2 = Conv3D(256, (5, 5, 4), padding='same')(prelu5_1)
    prelu5_2 = PReLU()(conv5_2)
    conv5_3 = Conv3D(256, (5, 5, 4), padding='same')(prelu5_2)
    prelu5_3 = PReLU()(conv5_3)
    add5 = Add()([prelu5_3, prelu4_4])
    upconv5 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2))(add5)
    prelu5_4 = PReLU()(upconv5)

    concat6 = Concatenate()([add4, prelu5_4])
    conv6_1 = Conv3D(256, (5, 5, 5), padding='same')(concat6)
    prelu6_1 = PReLU()(conv6_1)
    conv6_2 = Conv3D(256, (5, 5, 5), padding='same')(prelu6_1)
    prelu6_2 = PReLU()(conv6_2)
    conv6_3 = Conv3D(256, (5, 5, 5), padding='same')(prelu6_2)
    prelu6_3 = PReLU()(conv6_3)
    add6 = Add()([prelu6_3, concat6])
    upconv6 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2))(add6)
    prelu6_4 = PReLU()(upconv6)

    concat7 = Concatenate()([add3, prelu6_4])
    conv7_1 = Conv3D(128, (5, 5, 5), padding='same')(concat7)
    prelu7_1 = PReLU()(conv7_1)
    conv7_2 = Conv3D(128, (5, 5, 5), padding='same')(prelu7_1)
    prelu7_2 = PReLU()(conv7_2)
    conv7_3 = Conv3D(128, (5, 5, 5), padding='same')(prelu7_2)
    prelu7_3 = PReLU()(conv7_3)
    add7 = Add()([prelu7_3, concat7])
    upconv7 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2))(add7)
    prelu7_4 = PReLU()(upconv7)

    concat8 = Concatenate()([add2, prelu7_4])
    conv8_1 = Conv3D(64, (5, 5, 5), padding='same')(concat8)
    prelu8_1 = PReLU()(conv8_1)
    conv8_2 = Conv3D(64, (5, 5, 5), padding='same')(prelu8_1)
    prelu8_2 = PReLU()(conv8_2)
    add8 = Add()([prelu8_2, concat8])
    upconv8 = Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2))(add8)
    prelu8_3 = PReLU()(upconv8)

    concat9 = Concatenate()([add1, prelu8_3])
    conv9_1 = Conv3D(32, (5, 5, 5), padding='same')(concat9)
    prelu9_1 = PReLU()(conv9_1)
    add9 = Add()([prelu9_1, concat9])

    conv9_2 = Conv3D(1, (1, 1, 1), padding='same', activation='sigmoid')(add9)

    # softmax = Softmax()(conv9_2)
    # model = Model(inputs=[input_layer], outputs=[softmax])

    model = Model(input=[input_layer], outputs=[conv9_2])

    return model
