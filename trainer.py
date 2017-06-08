import tensorflow as tf
import tflearn
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import time

matplotlib.style.use('ggplot')


# Human Activity Recognition Using Smartphones Dataset
#
# [1] Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz.
# A Public Domain Dataset for Human Activity Recognition Using Smartphones.
# 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning,
# ESANN 2013. Bruges, Belgium 24-26 April 2013.
#
# http://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones
#
class HarData(object):
    max_timestep = 60
    output_size = 6

    @staticmethod
    def split_time(m, split_size):
        r = m.shape[0]
        extend_row_size = int(np.math.ceil((0.+r) / split_size)) * split_size - r
        print(extend_row_size)
        m_p = np.expand_dims(np.pad(m, [(0, extend_row_size), (0, 0)], mode='constant'), axis=0)
        result = m_p.reshape((int(np.math.ceil((0.+r) / split_size)), split_size, m.shape[1]))
        return result

    @staticmethod
    def convert(split_size=max_timestep):
        m = pd.read_csv('har_dataset/train/X_train.txt', header=None).as_matrix()
        x_train = HarData.split_time(m, split_size)
        m = pd.read_csv('har_dataset/train/y_train.txt', header=None).as_matrix()
        y_train = HarData.split_time(tflearn.data_utils.to_categorical(m - 1, HarData.output_size), split_size)
        m = pd.read_csv('har_dataset/test/X_test.txt', header=None).as_matrix()
        x_test = HarData.split_time(m, split_size)
        m = pd.read_csv('har_dataset/test/y_test.txt', header=None).as_matrix()
        y_test = HarData.split_time(tflearn.data_utils.to_categorical(m - 1, HarData.output_size), split_size)
        pd.to_pickle([(x_train, y_train), (x_test, y_test)], 'har_data.pkl')

    @staticmethod
    def load():
        return pd.read_pickle('har_data.pkl')


class Trainer(object):
    def __init__(self, max_timestep, feature_size, output_size):
        self._max_timestep = max_timestep
        self._feature_size = feature_size
        self._output_size = output_size
        self._hidden_size = 128
        self._batch_size = 16
        self._max_epoch = 100
        self._model_path = "har_model.ckpt"
        self._create_model()

    def _create_model(self):
        # [timestep, mini-batch, feature dims]
        self._x = tf.placeholder(tf.float32, [None, None, self._feature_size])
        self._y = tf.placeholder(tf.float32, [None, None, self._output_size])
        self._index = tf.placeholder(tf.int32, [None, ])

        initializer = tf.random_uniform_initializer(-1, 1)
        cell = tf.contrib.rnn.LSTMCell(self._hidden_size, self._feature_size, initializer=initializer)
        cell_out = tf.contrib.rnn.OutputProjectionWrapper(cell, self._output_size)
        outputs, _ = tf.nn.dynamic_rnn(cell_out, self._x, sequence_length=self._index, dtype=tf.float32,
                                       time_major=True)
        output_shape = tf.shape(outputs)
        prediction = tf.nn.softmax(tf.reshape(outputs, [-1, self._output_size]))
        self._prediction = tf.reshape(prediction, output_shape)
        self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf.reshape(outputs,[-1, self._output_size]),
                                                                      labels=tf.reshape(self._y, [-1, self._output_size])))
        #self._loss = tf.reduce_mean(tf.sqrt(tf.pow(self._prediction - self._y, 2)))  # mse
        self._optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self._loss)
        correct_prediction = tf.equal(tf.argmax(self._prediction, 2), tf.argmax(self._y, 2))
        self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def train(self, train_set, validation_set):
        init = tf.initialize_all_variables()
        saver = tf.train.Saver()
        train_total_size = train_set[0].shape[0]
        train_xs = train_set[0]
        train_ys = train_set[1]
        validation_total_size = validation_set[0].shape[0]
        validation_xs = np.swapaxes(validation_set[0], 0, 1)
        validation_ys = np.swapaxes(validation_set[1], 0, 1)
        train_total_batch = int((train_total_size+0.) / self._batch_size)

        # Launch the graph
        max_accuracy = 0.
        start_time = time.time()
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(self._max_epoch):
                avg_cost = 0.
                s = 0
                e = self._batch_size
                for i in range(train_total_batch):
                    sample_size = e - s
                    batch_xs = np.swapaxes(train_xs[s:e], 0, 1)
                    batch_ys = np.swapaxes(train_ys[s:e], 0, 1)
                    # TODO: adding argument for time indices
                    batch_indices = [self._max_timestep] * sample_size
                    # batch_indices = np.random.choice(max_timestep, sample_size)
                    sess.run(self._optimizer,
                             feed_dict={self._x: batch_xs, self._y: batch_ys, self._index: batch_indices})
                    cost = self._loss.eval(feed_dict={self._x: batch_xs, self._y: batch_ys, self._index: batch_indices})
                    avg_cost += (cost+0.) / train_total_batch
                    s = e
                    e += self._batch_size
                    if e > train_total_size:
                        e = train_total_size

                batch_indices = [self._max_timestep] * validation_total_size
                accuracy = self._accuracy.eval(
                    feed_dict={self._x: validation_xs, self._y: validation_ys, self._index: batch_indices})
                save_message = ""
                if accuracy > max_accuracy:
                    model_path = saver.save(sess, self._model_path)
                    max_accuracy = accuracy
                    save_message = "Model saved: {}".format(model_path)
                print("Epoch: {:03d}".format(epoch + 1), "Loss: {:.5f}".format(avg_cost),
                      "Accuracy: {:.3f}".format(accuracy), save_message)

            elapsed_time = time.time() - start_time
            print("elapsed_time:", elapsed_time)

    def test(self, test_set, index=0):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self._model_path)
            x = np.swapaxes(np.expand_dims(test_set[0][index], axis=0), 0, 1)
            y = np.swapaxes(np.expand_dims(test_set[1][index], axis=0), 0, 1)
            index = [x.shape[0]]
            y_hat = self._prediction.eval(feed_dict={self._x: x, self._y: y, self._index: index})
        return y, y_hat


def plot_output(y, y_hat):
    y = np.swapaxes(y, 0, 1)[0]
    y_hat = np.swapaxes(y_hat, 0, 1)[0]
    t = range(y.shape[0])
    plt.plot(t, y)
    plt.gca().set_prop_cycle(None)
    plt.plot(t, y_hat, ':', linewidth=2)
    plt.legend()
    plt.ylim([-0.1, 1.1])
    plt.show()


def main():
    train_set, test_set = HarData.load()
    print('Input shape:', train_set[0].shape)
    print('Output shape', train_set[1].shape)

    trainer = Trainer(HarData.max_timestep, train_set[0].shape[2], HarData.output_size)
    trainer.train(train_set, test_set)
    y, y_hat = trainer.test(test_set, 2)
    plot_output(y, y_hat)


if __name__ == '__main__':
    HarData.convert()
    main()
