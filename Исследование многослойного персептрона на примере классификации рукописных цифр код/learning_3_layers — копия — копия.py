from keras.datasets import mnist
import tensor as tf


(x_train, y_train), (x_test, y_test) = mnist.load_data()


images = tf.matrix3_to_2(x_train[0:1000])
answser = tf.answser_to_mass(y_train[0:1000])

iterations, hiden_size, pixels_per_image, num_labels = (100, 128, 784, 10)

alpha_1 = 0.001
alpha_2 = 2

weights0_1 = tf.spawn_weights(pixels_per_image, hiden_size,)
weights1_2 = tf.spawn_weights(hiden_size, num_labels, True)

drop_mark = 3  # каждый i элемент отключается
droping = False

for j in range(iterations):

  erro, correct_cnt = (0.0, 0)

  if j >= 4:
    drop_mark = 5

  if j == 5:
    droping = False


  for i in range(len(images)):
    layer_0 = images[i]
    layer_1 = tf.relu(tf.scal_mul(layer_0, weights0_1))
    layer_1 = tf.add_bias_to_layer(layer_1)

    if droping:
      drop = tf.dropout(layer_1, drop_mark, True)
      layer_1 = tf.scal_mul(tf.scal_mul(layer_1, drop, 3), 1 + (1 / drop_mark) * 2, 6)

    layer_2 = tf.softmax(tf.scal_mul(layer_1, weights1_2))

    erro += tf.mass_sum(tf.scal_mul(tf.error(answser[i], layer_2), tf.error(answser[i], layer_2), 3))
    if tf.mass_argmax(layer_2) == tf.mass_argmax(answser[i]):
      correct_cnt += 1

    layer_2_delta = tf.div(tf.error(answser[i], layer_2), len(layer_2))
    layer_1_delta = tf.scal_mul(tf.scal_mul(layer_2_delta, tf.reverse(weights1_2)), tf.relu2(layer_1), 3)

    if droping:
      layer_1_delta = tf.scal_mul(layer_1_delta, drop, 3)

    weights1_2_delta = tf.scal_mul(tf.scal_mul(layer_1, layer_2_delta, 5), alpha_2, 4)
    weights0_1_delta = tf.scal_mul(tf.scal_mul(layer_0, layer_1_delta, 5), alpha_1, 4)

    tf.matrix_sum(weights1_2, weights1_2_delta)
    tf.matrix_sum(weights0_1, weights0_1_delta)


    if i % 100 == 0:
      print("\r" + " I: " + str(j) + ' : ' + str(i) + " Error: " + str(erro / float(i + 1)) + "  Correct: " + str(correct_cnt / float(i + 1)))


file_weights0_1 = open('file3_5_weights0_1.py','w')
file_weights1_2 = open('file3_5_weights1_2.py','w')

file_weights0_1.write('weights0_1 = ' + str(weights0_1))
file_weights1_2.write('weights1_2 = ' + str(weights1_2))

file_weights0_1.close()
file_weights1_2.close()