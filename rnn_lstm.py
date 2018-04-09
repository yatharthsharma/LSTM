import numpy as np #matrix math
import tensorflow as tf #machine learningt
import helpers #for formatting data into batches and generating random sequence data
import helperForAutoencoder as h
import nltk

import sys
tf.reset_default_graph() #Clears the default graph stack and resets the global default graph.
sess = tf.InteractiveSession() #initializes a tensorflow session
tf.__version__
from tensorflow.python.ops.rnn_cell import LSTMCell, LSTMStateTuple
from tensorflow.python.ops import rnn
from util import init_weight, get_ptb_data, display_tree



def loop_fn_initial():
    initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
    #end of sentence
    initial_input = eos_step_embedded # 
    #last time steps cell state
    initial_cell_state = encoder_final_state
    #none
    initial_cell_output = None
    #none
    initial_loop_state = None  # we don't need to pass any additional information
    return (initial_elements_finished,
            initial_input,
            initial_cell_state,
            initial_cell_output,
            initial_loop_state)


    # attention mechanism --choose which previously generated token to pass as input in the next timestep
def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):
    def get_next_input():
        # dot product between previous ouput and weights, then + biases
        output_logits = tf.add(tf.matmul(previous_output, W), b)
        # Logits simply means that the function operates on the unscaled output of
        # earlier layers and that the relative scale to understand the units is linear.
        # It means, in particular, the sum of the inputs may not equal 1, that the values are not probabilities
        # (you might have an input of 5).
        # prediction value at current time step

        # Returns the index with the largest value across axes of a tensor.
        prediction = tf.argmax(output_logits, 1)
        # embed prediction for the next input
        next_input = tf.nn.embedding_lookup(We, prediction)
        return next_input

    elements_finished = (time >= decoder_lengths)  # this operation produces boolean tensor of [batch_size]
    # defining if corresponding sequence has ended

    # Computes the "logical and" of elements across dimensions of a tensor.



    finished = tf.reduce_all(elements_finished)  # -> boolean scalar
    # Return either fn1() or fn2() based on the boolean predicate pred.
    input = tf.cond(finished, lambda: pad_step_embedded, get_next_input)



    # set previous to current
    state = previous_state
    output = previous_output
    loop_state = None

    return (elements_finished,
            input,
            state,
            output,
            loop_state)


def loop_fn(time, previous_output, previous_state, previous_loop_state):
    if previous_state is None:    # time == 0
        assert previous_output is None and previous_state is None
        return loop_fn_initial()
    else:
        return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)


def get_output_recursive( tree, list_of_logits, wordEmb_list, is_root=True):
    if tree.word is not None:
        # this is a leaf node
        x = tf.nn.embedding_lookup(We, [tree.word])
# 
    else:
        # this node has children
        x1 = get_output_recursive(tree.left, list_of_logits, wordEmb_list, is_root=False)
        x2 = get_output_recursive(tree.right, list_of_logits, wordEmb_list, is_root=False)
        x = f(
            tf.matmul(x1, W1) +
            tf.matmul(x2, W2) +
            bh)

    logits = tf.matmul(x, Wo) + bo
    wordEmb_list.append(x)
    list_of_logits.append(logits)
    # print tf.Print(x,[x])
    # print ("-------------------------------")
    return x

def get_output( tree):
    logits = []
    wordEmb = []

    # try:
    get_output_recursive(tree, logits, wordEmb)
    # except Exception as e:
    # display_tree(tree)

    #     raise e
    return (tf.concat(0,logits),wordEmb)



#datasetPath = "datasetSentences.txt"
# datasetPath = "trees/train.txt"
# lookUp, reverseLookUp = h.getLookUps(datasetPath)

# sentences=h.getSentences(datasetPath,lookUp)
# print("vocab size: "+str(len(lookUp))+" number of sentences: "+str(len(sentences)))
train, test, word2idx = get_ptb_data()

train = train[:20]
test = test[:1]

V = len(word2idx)
D = 20
K = 5

# model = TNN(V, D, K, tf.nn.relu)
# model.fit(train)

D = D
f = tf.nn.relu

# word embedding
# We = init_weight(V, D)
We = tf.Variable(tf.random_uniform([V, D], -1.0, 1.0), dtype=tf.float32)
# linear terms
W1 = init_weight(D, D)
W2 = init_weight(D, D)

# bias
bh = np.zeros(D)

# output layer
Wo = init_weight(D, D)
bo = np.zeros(D)

# make them tensorflow variables
# We = tf.Variable(We.astype(np.float32))
W1 = tf.Variable(W1.astype(np.float32))
W2 = tf.Variable(W2.astype(np.float32))
bh = tf.Variable(bh.astype(np.float32))
Wo = tf.Variable(Wo.astype(np.float32))
bo = tf.Variable(bo.astype(np.float32))
params = [We, W1, W2, Wo]


PAD = 0
EOS = 1

datasetPath = "trees/train_sent.txt"
lookUp, reverseLookUp = h.getLookUps(datasetPath)

sentences=h.getSentences(datasetPath,lookUp)
print("vocab size: "+str(len(lookUp))+" number of sentences: "+str(len(sentences)))

encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')
decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')

batch = h.createBatch(sentences, 0, 20)

encoder_inputs, encoder_input_lengths = helpers.batch(batch)
decoder_targets, _ = helpers.batch(
    [(sequence) + [EOS] + [PAD] * 2 for sequence in batch]
)

# print encoder_inputs

vocab_size = V
# input_embedding_size = 300 #character length

# encoder_hidden_units = 20 #num neurons
decoder_hidden_units = 20 

# encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')

encoder_inputs_embedded = tf.nn.embedding_lookup(We, encoder_inputs)
#figute this out

# same as We


# this thing could get huge in a real world application
# encoder_inputs_embedded = tf.nn.embedding_lookup(We, encoder_inputs)

# # encoder_cell = LSTMCell(encoder_hidden_units)


# ((encoder_fw_outputs,
#   encoder_bw_outputs),
#  (encoder_fw_final_state,
#   encoder_bw_final_state)) = (
#     tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
#                                     cell_bw=encoder_cell,
#                                     inputs=encoder_inputs_embedded,
#                                     sequence_length=encoder_inputs_length,
#                                     dtype=tf.float32, time_major=True)
#     )
# encoder_fw_outputs
# encoder_bw_outputs
# encoder_fw_final_state

# encoder_bw_final_state
# #Concatenates tensors along one dimension.
# encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)


#fit func!!!
# def fit( trees, lr=10e-3, mu=0, reg=10e-3, epochs=3):



train_ops = []
costs = []
predictions = []
all_labels = []
i = 0
trees = train
N = len(trees)

print("Compiling ops")

for t in trees:
# print (t.word)  

    i += 1
    sys.stdout.write("%d/%d\r" % (i, N))
    sys.stdout.flush()
    logits, wordEmb = get_output(t)


    encoder_outputs = logits[-1]

    state = (tf.zeros([20,1]),)

    encoder_final_state_c = tf.zeros([decoder_hidden_units,decoder_hidden_units],dtype=tf.float32)

    encoder_final_state_h = tf.zeros([decoder_hidden_units,decoder_hidden_units],dtype=tf.float32)

    #TF Tuple used by LSTM Cells for state_size, zero_state, and output state.

    encoder_final_state = LSTMStateTuple(
        c=encoder_final_state_c,
        h=encoder_final_state_h
    )


    decoder_cell = LSTMCell(decoder_hidden_units,state_is_tuple=False)
    #we could print this, won't need
    # decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

    # encoder_max_time, batch_size = tf.unstack(tf.shape(encoder_inputs))
    decoder_lengths = encoder_inputs_length[i] + 3
    # # +2 additional steps, +1 leading <EOS> token for decoder inputs
    # #manually specifying since we are going to implement attention details for the decoder in a sec
    # #weights


    W = tf.Variable(tf.random_uniform([decoder_hidden_units, vocab_size], -1, 1), dtype=tf.float32)
    #bias
    b = tf.Variable(tf.zeros([vocab_size]), dtype=tf.float32)
    #create padded inputs for the decoder from the word We

    #were telling the program to test a condition, and trigger an error if the condition is false.
    # assert EOS == 1 and PAD == 0

    eos_time_slice = tf.ones([20], dtype=tf.int32, name='EOS') #change this to change padding and eos values
    pad_time_slice = tf.zeros([20], dtype=tf.int32, name='PAD')

    #retrieves rows of the params tensor. The behavior is similar to using indexing with arrays in numpy
    # eos_step_embedded = tf.nn.embedding_lookup(We, eos_time_slice)

    eos_step_embedded =  tf.nn.embedding_lookup(We, eos_time_slice)

    pad_step_embedded = tf.nn.embedding_lookup(We, pad_time_slice)


    # decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
    # decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
    outputs, state = tf.nn.dynamic_rnn(decoder_cell,  tf.nn.embedding_lookup(We, encoder_inputs),
                                   dtype=tf.float32)
    # outputs, state = tf.nn.dynamic_rnn(decoder_cell,encoder_inputs,sequence_length=[10]*19,dtype=tf.float32)
    # decoder_outputs_ta, decoder_final_state, _ = rnn.rnn(decoder_cell,logits,dtype=tf.float32)
    decoder_outputs = decoder_outputs_ta.stack()


    decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
    #flettened output tensor
    decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
    #pass flattened tensor through decoder
    decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)
    #prediction vals
    decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, vocab_size))
    #final prediction
    decoder_prediction = tf.argmax(decoder_logits, 2)
    #cross entropy loss
    #one hot encode the target values so we don't rank just differentiate
    stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
        logits=decoder_logits,
    )

    #loss function
    loss = tf.reduce_mean(stepwise_cross_entropy)
    #train it
    train_op = tf.train.AdamOptimizer().minimize(loss)

batch_size = 100            

# saver = tf.train.Saver()

actual_costs = []
per_epoch_costs = []
correct_rates = []
with tf.Session() as session:
    sess.run(tf.global_variables_initializer())
    # writer = tf.train.SummaryWriter('logs', graph=tf.get_default_graph())
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
            # print (len(session.run(Wo[:,0])))

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

    # saver.save(session, "recursive.ckpt")
    # writer.close()


    # def next_feed(start):
    #     batch = h.createBatch(sentences, start, batch_size)

    #     encoder_inputs_, encoder_input_lengths_ = helpers.batch(batch)
    #     decoder_targets_, _ = helpers.batch(
    #         [(sequence) + [EOS] + [PAD] * 2 for sequence in batch]
    #     )
    #     return {
    #         encoder_inputs: encoder_inputs_,
    #         encoder_inputs_length: encoder_input_lengths_,
    #         decoder_targets: decoder_targets_,
    #     }

    # loss_track = []
    # max_batches = int(len(sentences)/batch_size)
    # batches_in_epoch = 100

    # iteration=100
    # flag=0
    # try:
    #     while True:
    #         start=0
    #         for batch in range(max_batches):
    #             if start> len(sentences):
    #                 break

    #             fd = next_feed(start)

    #             _, l = sess.run([train_op, loss], fd)
    #             loss_track.append(l)

    #             if batch == 0 or batch % batches_in_epoch == 0:
    #                 mLoss= sess.run(loss,fd)
    #                 if mLoss < 2:
    #                     flag=1
    #                     break
    #                 print('batch {}'.format(batch))
    #                 print('  minibatch loss: {}'.format(mLoss))
    #                 predict_ = sess.run(decoder_prediction, fd)
    #                 for i, (inp, pred) in enumerate(zip(fd[encoder_inputs].T, predict_.T)):
    #                     print('  sample {}:'.format(i + 1))
    #                     print('    input     > {}'.format(h.toWords(inp,reverseLookUp)))
    #                     print('    predicted > {}'.format(h.toWords(pred,reverseLookUp)))
    #                     print('    BLEUScore > '+str(nltk.translate.bleu_score.sentence_bleu([h.toWords(inp,reverseLookUp)], h.toWords(pred,reverseLookUp))))
    #                     if i >= 2:
    #                         break
    #                 print()
    #             start=start+batch_size
    #         if flag==1:
    #             break

    # except KeyboardInterrupt:
    #     print('training interrupted')


