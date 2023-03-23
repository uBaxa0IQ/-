from keras.datasets import mnist
import tensor as tf


(x_train, y_train),(x_test, y_test) = mnist.load_data()



#images = tf.matrix3_to_2(tf.shiiiii(tf.shiiiii(tf.shiiiii(x_train[0:20000] , tf.change_matrix(x_train[0:30000])) , x_train[20000:60000]), tf.duble_change_matrix(x_train[30000:60000])))
#answser = tf.answser_to_mass(tf.shiiiii(tf.shiiiii(tf.shiiiii(y_train[0:20000] , (y_train[0:30000])) , y_train[20000:60000]), y_train[30000:60000]))

images = tf.matrix3_to_2(x_train[0:60000])
answser = tf.answser_to_mass(y_train[0:60000])

iterations, hiden_size_1, hiden_size_2, pixels_per_image , num_labels = (5, 250, 80, 784, 10)

alpha_1 = 0.0001
alpha_2 = 0.0001
alpha_3 = 0.1
droping = True
drop_mark = 3

weights0_1 = tf.spawn_weights(pixels_per_image, hiden_size_1)
weights1_2 = tf.spawn_weights(hiden_size_1, hiden_size_2)
weights2_3 = tf.spawn_weights(hiden_size_2, num_labels)

print('start')

for j in range(iterations):

  erro, correct_cnt = (0.0, 0)
  correct_cnt_1 = 0

  if j >= 0.6 * iterations:
    drop_mark = 5

  if j / iterations >= 0.8:
    droping = False

  for i in range(len(images)):
    layer_0 = images[i]
    layer_1 = tf.relu(tf.scal_mul(layer_0, weights0_1))

    if droping:
      drop = tf.dropout(layer_1, drop_mark, True)
      layer_1 = tf.scal_mul(tf.scal_mul(layer_1, drop, 3), 1 + (1 / drop_mark) * 2, 6)

    layer_2 = tf.relu(tf.scal_mul(layer_1, weights1_2))

    layer_3 = tf.softmax(tf.scal_mul(layer_2, weights2_3))

    erro += tf.mass_sum(tf.scal_mul(tf.error(answser[i], layer_3), tf.error(answser[i], layer_3), 3))
    if tf.mass_argmax(layer_3) == tf.mass_argmax(answser[i]):
      correct_cnt += 1
      correct_cnt_1 += 1

    layer_3_delta = tf.div(tf.error(answser[i], layer_3), len(layer_3))
    layer_2_delta = tf.scal_mul(tf.scal_mul(layer_3_delta, tf.reverse(weights2_3)), tf.relu2(layer_2), 3)
    layer_1_delta = tf.scal_mul(tf.scal_mul(layer_2_delta, tf.reverse(weights1_2)), tf.relu2(layer_1), 3)

    if droping:
      layer_1_delta = tf.scal_mul(layer_1_delta, drop, 3)

    weights2_3_delta = tf.scal_mul(tf.scal_mul(layer_2, layer_3_delta, 5), alpha_3, 4)
    weights1_2_delta = tf.scal_mul(tf.scal_mul(layer_1, layer_2_delta, 5), alpha_1, 4)
    weights0_1_delta = tf.scal_mul(tf.scal_mul(layer_0, layer_1_delta, 5), alpha_1, 4)

    tf.matrix_sum(weights2_3, weights2_3_delta)
    tf.matrix_sum(weights1_2, weights1_2_delta)
    tf.matrix_sum(weights0_1, weights0_1_delta)

    if i % 100 == 0 and i != 0:
      print("\r" + " I: " + str(j) + ' : ' + str(i) + " Error: " + str(erro / float(i + 1)) + "  Correct: " + str(
        correct_cnt / float(i + 1)))
      #if (i % 500 == 0 and i != 0):
        #graf_1.append(erro / 500)
        #graf_2.append(correct_cnt_1 / 500)
        # plt.plot(graf_1)
        # plt.plot(graf_2)
        #plt.plot(graf_2)
        #plt.show()
        #correct_cnt_1 = 0


file_weights0_1 = open('file4_7_weights0_1.py','w')
file_weights1_2 = open('file4_7_weights1_2.py','w')
file_weights2_3 = open('file4_7_weights2_3.py','w')

file_weights0_1.write('weights0_1 = ' + str(weights0_1))
file_weights1_2.write('weights1_2 = ' + str(weights1_2))
file_weights2_3.write('weights2_3 = ' + str(weights2_3))

file_weights0_1.close()
file_weights1_2.close()
file_weights2_3.close()