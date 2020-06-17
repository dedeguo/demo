import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# import tensorflow_datasets as tfds
# tfds.disable_progress_bar()

embedding_layer = layers.Embedding(1000, 5)
result = embedding_layer(tf.constant([1,2,3,4]))
print(result.shape)

message = tf.constant('Welcome to the exciting world of Deep Neural Networks!')

with tf.Session() as sess:
    print(sess.run(message).decode())
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # 隐式指定了tensor内部数据的类型是tf.float32
print(node1, node2)