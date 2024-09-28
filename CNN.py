import tensorflow as tf
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, Dense, Flatten, Input
from tensorflow.keras.models import Model


# 构建CNN来替代原来的DNN
def build_cnn(input_shape):
    inputs = Input(shape=input_shape)

    # 第一卷积层
    x = Conv1D(64, kernel_size=3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 第二卷积层
    x = Conv1D(128, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 第三卷积层
    x = Conv1D(256, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 扁平化层
    x = Flatten()(x)

    # 全连接层，输出16个比特
    output = Dense(16, activation='sigmoid')(x)

    # 构建模型
    model = Model(inputs=inputs, outputs=output)

    return model


# 输入：128个复数 -> 128个复数分成实部和虚部（即256个输入值）
input_shape = (128, 2)  # 输入形状为 (128, 2)，128个复数的实部和虚部
model = build_cnn(input_shape)
model.summary()
