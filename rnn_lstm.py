import numpy as np #matrix math
import tensorflow as tf #machine learningt
import helpers #for formatting data into batches and generating random sequence data
import helperForAutoencoder as h
import nltk
from sklearn.utils import shuffle
import sys
import tflearn
tf.reset_default_graph() #Clears the default graph stack and resets the global default graph.
sess = tf.InteractiveSession() #initializes a tensorflow session
tf.__version__
from tensorflow.python.ops.rnn_cell import LSTMCell, LSTMStateTuple
from tensorflow.python.ops import rnn
from util import init_weight, get_ptb_data, display_tree



def loop_fn_initial():
    initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
    #end of sentence
    initial_input = eos_step_embedded
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
        prediction = tf.argmax(output_logits,1)
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
    return logits

train, test, word2idx = get_ptb_data()

train = train[:1]
test = test[:1]

V = len(word2idx)
D = 20
K = 5

D = D
f = tf.nn.relu

# word embedding
We = init_weight(V, D)
# We = tf.Variable(tf.random_uniform([V, D], -1.0, 1.0), dtype=tf.float32)
# linear terms
W1 = init_weight(D, D)
W2 = init_weight(D, D)

# bias
bh = np.zeros(D)

# output layer
Wo = init_weight(D, D)
bo = np.zeros(D)

# make them tensorflow variables
We = tf.Variable(We.astype(np.float32),name='WordEmb')
W1 = tf.Variable(W1.astype(np.float32),name='W1')
W2 = tf.Variable(W2.astype(np.float32),name='W2')
bh = tf.Variable(bh.astype(np.float32),name='bh')
Wo = tf.Variable(Wo.astype(np.float32),name='Wo')
bo = tf.Variable(bo.astype(np.float32),name='Bo')
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

encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')

encoder_inputs_embedded = tf.nn.embedding_lookup(We, encoder_inputs)
#figute this out

train_ops = []
costs = []
predictions = []
all_labels = []
i = 0
trees = train
N = len(trees)
encoder_outputs = []
print("Compiling ops")

for t in trees:
# print (t.word)  

    i += 1
    sys.stdout.write("%d/%d\r" % (i, N))
    sys.stdout.flush()
    logits = get_output(t)

    # encoder_outputs.append(logits[-1])
    encoder_outputs = logits[-1]

    # encoder_outputs.append(logits[-1])

# state = tf.zeros([1,20])

# encoder_final_state_c = tf.zeros([2,],dtype=tf.float32)

# encoder_final_state_h = tf.zeros([2,],dtype=tf.float32)

# #TF Tuple used by LSTM Cells for state_size, zero_state, and output state.

    input_series = [tf.reshape(encoder_outputs,[-1,20])]

    # input_series = [tf.reshape(ipt,[-1,20]) for ipt in encoder_outputs]
    # encoder_outputs = tf.reshape(encoder_outputs,[0, 20])
    encoder_final_state_c= tf.placeholder(tf.float32, shape=[None,20])
    encoder_final_state_h = tf.placeholder(tf.float32, shape=[None,20])


    encoder_final_state = LSTMStateTuple(
        c=encoder_outputs,
        h=encoder_outputs
    )
    # decoder_cell = BasesLSTMCell(decoder_hidden_units, state_is_tuple=True) 
    decoder_cell = LSTMCell(decoder_hidden_units, state_is_tuple=True)
    # decoder_cell = LSTMCell(decoder_hidden_units)
    states_series, current_state = tf.nn.static_rnn(decoder_cell, input_series, encoder_final_state) 
#we could print this, won't need
# decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(decoder_hidden_units)
# encoder_final_state_cencoder_outputs
# encoder_max_time, batch_size = tf.unstack(tf.shape(encoder_inputs))
# decoder_lengths = encoder_inputs_length + 3



# W = tf.Variable(tf.random_uniform([decoder_hidden_units, vocab_size], -1, 1), dtype=tf.float32)
# # #bias
# b = tf.Variable(tf.zeros([vocab_size]), dtype=tf.float32)
# # #create padded inputs for the decoder from the word We

# # #were telling the program to test a condition, and trigger an error if the condition is false.
# # # assert EOS == 1 and PAD == 0

# eos_time_slice = tf.ones([20], dtype=tf.int32, name='EOS') #change this to change padding and eos values
# pad_time_slice = tf.zeros([20], dtype=tf.int32, name='PAD')

# # #retrieves rows of the params tensor. The behavior is similar to using indexing with arrays in numpy
# # eos_step_embedded = tf.nn.embedding_lookup(We, eos_time_slice)

# eos_step_embedded =  tf.nn.embedding_lookup(We, eos_time_slice)

# pad_step_embedded = tf.nn.embedding_lookup(We, pad_time_slice)

# decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
# decoder_outputs_ta
# decoder_outputs = decoder_outputs_ta.stack()

# decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
# decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
# decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)
# decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, vocab_size))
# decoder_prediction = tf.argmax(decoder_logits, 2)
# stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
#     labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
#     logits=decoder_logits,
# )

# #loss function
# loss = tf.reduce_mean(stepwise_cross_entropy)
#train it
# train_op = tf.train.AdamOptimizer().minimize(loss)

batch_size = 100            

# saver = tf.train.Saver()
epochs =10
actual_costs = []
per_epoch_costs = []
correct_rates = []
# init = tf.initialize_all_variables()
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    # sess.run(init)
    writer = tf.summary.FileWriter('logs', graph=tf.get_default_graph())
    # writer = tf.train.SummaryWriter('logs', graph=tf.get_default_graph())
    for i in range(epochs):
        # t0 = datetime.now()

        train_ops, costs, predictions = shuffle(train_ops, costs, predictions)
        epoch_cost = 0
        n_correct = 0
        n_total = 0
        j = 0
        N = len(train_ops)
        for train_op, cost, prediction, labels in zip(train_ops, costs, predictions):
            t_op, c, p = session.run([train_op,loss,decoder_prediction])
            # print("train ops")

      
            epoch_cost += c
            actual_costs.append(c)
            n_correct += np.sum(p == labels)
            n_total += len(labels)

            # j += 1
            # if j % 10 == 0:
            #     sys.stdout.write("j: %d, N: %d, c: %f\r" % (j, N, c))
            #     sys.stdout.flush()

        # print(
        #     "epoch:", i, "cost:", epoch_cost,
        #     "elapsed time:", (datetime.now() - t0)
        # )

        # per_epoch_costs.append(epoch_cost)
        # correct_rates.append(n_correct / float(n_total))

    writer.close()
