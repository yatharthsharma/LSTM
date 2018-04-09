from __future__ import print_function, division
from builtins import range


import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from datetime import datetime
from util import init_weight, get_ptb_data, display_tree
from tensorflow.python.ops.rnn_cell import LSTMCell, LSTMStateTuple
import helpers  
class TNN:
    def __init__(self, V, D, K, activation):
        self.D = D
        self.f = activation

        # word embedding
        We = init_weight(V, D)

        # linear terms
        W1 = init_weight(D, D)
        W2 = init_weight(D, D)

        # bias
        bh = np.zeros(D)

        # output layer
        Wo = init_weight(D, D)
        bo = np.zeros(D)

        # make them tensorflow variables
        self.We = tf.Variable(We.astype(np.float32))
        self.W1 = tf.Variable(W1.astype(np.float32))
        self.W2 = tf.Variable(W2.astype(np.float32))
        self.bh = tf.Variable(bh.astype(np.float32))
        self.Wo = tf.Variable(Wo.astype(np.float32))
        self.bo = tf.Variable(bo.astype(np.float32))
        self.params = [self.We, self.W1, self.W2, self.Wo]

    def fit(self, trees, lr=10e-3, mu=0, reg=10e-3, epochs=3):
        train_ops = []
        costs = []
        predictions = []
        all_labels = []
        i = 0
        N = len(trees)

        print("Compiling ops")


        for t in trees:
            # print (t.word)  
            i += 1
            sys.stdout.write("%d/%d\r" % (i, N))
            sys.stdout.flush()
            logits, wordEmb = self.get_output(t)
        
            decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(200)


            helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, decoder_lengths, time_major=True)

            decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, encoder_state,output_layer=projection_layer)
            outputs, _ = tf.contrib.seq2seq.dynamic_decode(decode)
            logitsF =    outputs.rnn_output
        
        self.saver = tf.train.Saver()

        init = tf.initialize_all_variables()
        actual_costs = []
        per_epoch_costs = []
        correct_rates = []
        with tf.Session() as session:
            session.run(init)

            # writer = tf.summary.FileWriter('../logs', session.graph)
            writer = tf.train.SummaryWriter('logs', graph=tf.get_default_graph())
            for i in range(epochs):
                t0 = datetime.now()

                train_ops, costs, predictions, all_labels = shuffle(train_ops, costs, predictions, all_labels)
                epoch_cost = 0
                n_correct = 0
                n_total = 0
                j = 0
                N = len(train_ops)
                for train_op, cost, prediction, labels in zip(train_ops, costs, predictions, all_labels):
                    print("---------------")
                    t_op, c, p = session.run([train_op, cost, prediction])
                    # print("train ops")
                    print (len(p))
                    print("len--->")
                    print (len(session.run(logits)))
                    # print (len(session.run(self.Wo[:,0])))

                    epoch_cost += c
                    actual_costs.append(c)
                    n_correct += np.sum(p == labels)
                    n_total += len(labels)

                    j += 1
                    if j % 10 == 0:
                        sys.stdout.write("j: %d, N: %d, c: %f\r" % (j, N, c))
                        sys.stdout.flush()

                print(
                    "epoch:", i, "cost:", epoch_cost,
                    "elapsed time:", (datetime.now() - t0)
                )

                per_epoch_costs.append(epoch_cost)
                correct_rates.append(n_correct / float(n_total))

            # variables_names =[v.name for v in tf.trainable_variables()]

            # for k,v in zip(variables_names, values):
                # print(k, v)

            self.saver.save(session, "recursive.ckpt")
            writer.close()
   
    def get_output_recursive(self, tree, list_of_logits, wordEmb_list, is_root=True):
        if tree.word is not None:
            # this is a leaf node
            x = tf.nn.embedding_lookup(self.We, [tree.word])
# 
        else:
            # this node has children
            x1 = self.get_output_recursive(tree.left, list_of_logits, wordEmb_list, is_root=False)
            x2 = self.get_output_recursive(tree.right, list_of_logits, wordEmb_list, is_root=False)
            x = self.f(
                tf.matmul(x1, self.W1) +
                tf.matmul(x2, self.W2) +
                self.bh)

        logits = tf.matmul(x, self.Wo) + self.bo
        wordEmb_list.append(x)
        list_of_logits.append(logits)
        # print tf.Print(x,[x])
        # print ("-------------------------------")
        return x

    def get_output(self, tree):
        logits = []
        wordEmb = []

        # try:
        self.get_output_recursive(tree, logits, wordEmb)
        # except Exception as e:
        # display_tree(tree)

        #     raise e
        return (tf.concat(0,logits),wordEmb)

 

def main():
    train, test, word2idx = get_ptb_data()

    train = train[:1]
    test = test[:1]

    V = len(word2idx)
    D = 20
    K = 5

    model = TNN(V, D, K, tf.nn.relu)
    model.fit(train)
    # print("train accuracy:", model.score(None))
    # print("test accuracy:", model.score(test))


if __name__ == '__main__':
    main()