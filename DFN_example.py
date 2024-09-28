import tensorflow as tf
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, Add, Dense, Flatten, Input
from tensorflow.keras.models import Model


# 残差块定义
def residual_block(input_tensor, filters, kernel_size=3):
    # 第一卷积层
    x = Conv1D(filters=filters, kernel_size=kernel_size, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 第二卷积层
    x = Conv1D(filters=filters, kernel_size=kernel_size, padding='same')(x)
    x = BatchNormalization()(x)

    # 残差连接：将输入与卷积后的输出相加
    x = Add()([x, input_tensor])
    x = Activation('relu')(x)

    return x


# 构建带残差连接的CNN模型
def build_resnet_cnn(input_shape):
    inputs = Input(shape=input_shape)

    # 初始卷积层
    x = Conv1D(64, kernel_size=3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 第一个残差块
    x = residual_block(x, 64)

    # 第二个残差块
    x = residual_block(x, 64)

    # 扁平化
    x = Flatten()(x)

    # 全连接层，输出16个比特
    x = Dense(16, activation='sigmoid')(x)

    # 构建模型
    model = Model(inputs=inputs, outputs=x)

    return model


# 输入：128个复数 -> 128个复数分成实部和虚部（即256个输入值）
input_shape = (128, 2)  # 128个复数可以表示为实部和虚部，即形状为 (128, 2)
model = build_resnet_cnn(input_shape)
model.summary()
