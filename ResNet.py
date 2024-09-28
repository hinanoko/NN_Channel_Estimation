import tensorflow as tf


def encoder_with_residual(x):
    weights = {
        'encoder_h1': tf.Variable(tf.random.truncated_normal([n_input, n_hidden_1], stddev=0.1)),
        'encoder_h2': tf.Variable(tf.random.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.1)),
        'encoder_h3': tf.Variable(tf.random.truncated_normal([n_hidden_2, n_hidden_3], stddev=0.1)),
        'encoder_h4': tf.Variable(tf.random.truncated_normal([n_hidden_3, n_output], stddev=0.1)),
    }

    biases = {
        'encoder_b1': tf.Variable(tf.random.truncated_normal([n_hidden_1], stddev=0.1)),
        'encoder_b2': tf.Variable(tf.random.truncated_normal([n_hidden_2], stddev=0.1)),
        'encoder_b3': tf.Variable(tf.random.truncated_normal([n_hidden_3], stddev=0.1)),
        'encoder_b4': tf.Variable(tf.random.truncated_normal([n_output], stddev=0.1)),
    }

    # 第一层
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))

    # 第二层 + 残差连接
    layer_2 = tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2'])
    residual_output_2 = tf.add(layer_2, layer_1)  # 残差连接
    layer_2_activated = tf.nn.relu(residual_output_2)  # 激活函数

    # 第三层 + 残差连接
    layer_3 = tf.add(tf.matmul(layer_2_activated, weights['encoder_h3']), biases['encoder_b3'])
    residual_output_3 = tf.add(layer_3, layer_2_activated)  # 残差连接
    layer_3_activated = tf.nn.relu(residual_output_3)  # 激活函数

    # 第四层 (无残差连接，作为输出层)
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3_activated, weights['encoder_h4']), biases['encoder_b4']))

    return layer_4
