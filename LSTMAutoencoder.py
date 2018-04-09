#we give encoder input sequence like 'hello how are you', we take the last hidden state and feed to decoder and it
#will generate a decoded value. we compare that to target value, if translation would be 'bonjour ca va' and minimize
#the difference by optimizing a loss function

#in this case we just want to encode and decode the input successfully

#bidirectional encoder
#We will teach our model to memorize and reproduce input sequence.
#Sequences will be random, with varying length.
#Since random sequences do not contain any structure,
#model will not be able to exploit any patterns in data.
#It will simply encode sequence in a thought vector, then decode from it.
#this is not about prediction (end goal), it's about understanding this architecture

#this is an encoder-decoder architecture. The encoder is bidrectional so
#it It feeds previously generated tokens during training as inputs, instead of target sequence.

import numpy as np #matrix math
import tensorflow as tf #machine learningt
import helpers #for formatting data into batches and generating random sequence data
import helperForAutoencoder as h
import nltk
tf.reset_default_graph() #Clears the default graph stack and resets the global default graph.
sess = tf.InteractiveSession() #initializes a tensorflow session
tf.__version__
#First critical thing to decide: vocabulary size.
#Dynamic RNN models can be adapted to different batch sizes
#and sequence lengths without retraining
#(e.g. by serializing model parameters and Graph definitions via tf.train.Saver),
#but changing vocabulary size requires retraining the model.

#datasetPath = "datasetSentences.txt"
datasetPath = "hp.txt"
lookUp, reverseLookUp = h.getLookUps(datasetPath)

sentences=h.getSentences(datasetPath,lookUp)
print("vocab size: "+str(len(lookUp))+" number of sentences: "+str(len(sentences)))



PAD = 0
EOS = 1

vocab_size = len(lookUp)
input_embedding_size = 300 #character length

encoder_hidden_units = 20 #num neurons
decoder_hidden_units = encoder_hidden_units * 2 #in original paper, they used same number of neurons for both encoder
#and decoder, but we use twice as many so decoded output is different, the target value is the original input
#in this example
#input placehodlers
encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
#contains the lengths for each of the sequence in the batch, we will pad so all the same
#if you don't want to pad, check out dynamic memory networks to input variable length sequences
encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')
decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
#randomly initialized embedding matrrix that can fit input sequence
#used to convert sequences to vectors (embeddings) for both encoder and decoder of the right size
#reshaping is a thing, in TF you gotta make sure you tensors are the right shape (num dimensions)
embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)

#this thing could get huge in a real world application
encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
from tensorflow.python.ops.rnn_cell import LSTMCell, LSTMStateTuple
encoder_cell = LSTMCell(encoder_hidden_units)
#get outputs and states
#bidirectional RNN function takes a separate cell argument for
#both the forward and backward RNN, and returns separate
#outputs and states for both the forward and backward RNN

#When using a standard RNN to make predictions we are only taking the “past” into account.
#For certain tasks this makes sense (e.g. predicting the next word), but for some tasks
#it would be useful to take both the past and the future into account. Think of a tagging task,
#like part-of-speech tagging, where we want to assign a tag to each word in a sentence.
#Here we already know the full sequence of words, and for each word we want to take not only the
#words to the left (past) but also the words to the right (future) into account when making a prediction.
#Bidirectional RNNs do exactly that. A bidirectional RNN is a combination of two RNNs – one runs forward from
#“left to right” and one runs backward from “right to left”. These are commonly used for tagging tasks, or
#when we want to embed a sequence into a fixed-length vector (beyond the scope of this post).


((encoder_fw_outputs,
  encoder_bw_outputs),
 (encoder_fw_final_state,
  encoder_bw_final_state)) = (
    tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                    cell_bw=encoder_cell,
                                    inputs=encoder_inputs_embedded,
                                    sequence_length=encoder_inputs_length,
                                    dtype=tf.float32, time_major=True)
    )
encoder_fw_outputs
encoder_bw_outputs
encoder_fw_final_state

encoder_bw_final_state
#Concatenates tensors along one dimension.
encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

#letters h and c are commonly used to denote "output value" and "cell state".
#http://colah.github.io/posts/2015-08-Understanding-LSTMs/
#Those tensors represent combined internal state of the cell, and should be passed together.

encoder_final_state_c = tf.concat(
    (encoder_fw_final_state.c, encoder_bw_final_state.c), 1)

encoder_final_state_h = tf.concat(
    (encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

#TF Tuple used by LSTM Cells for state_size, zero_state, and output state.
encoder_final_state = LSTMStateTuple(
    c=encoder_final_state_c,
    h=encoder_final_state_h
)
decoder_cell = LSTMCell(decoder_hidden_units)
#we could print this, won't need
encoder_max_time, batch_size = tf.unstack(tf.shape(encoder_inputs))
decoder_lengths = encoder_inputs_length + 3
# +2 additional steps, +1 leading <EOS> token for decoder inputs
#manually specifying since we are going to implement attention details for the decoder in a sec
#weights
W = tf.Variable(tf.random_uniform([decoder_hidden_units, vocab_size], -1, 1), dtype=tf.float32)
#bias
b = tf.Variable(tf.zeros([vocab_size]), dtype=tf.float32)
#create padded inputs for the decoder from the word embeddings

#were telling the program to test a condition, and trigger an error if the condition is false.
assert EOS == 1 and PAD == 0

eos_time_slice = tf.ones([batch_size], dtype=tf.int32, name='EOS') #change this to change padding and eos values
pad_time_slice = tf.zeros([batch_size], dtype=tf.int32, name='PAD')

#retrieves rows of the params tensor. The behavior is similar to using indexing with arrays in numpy
eos_step_embedded = tf.nn.embedding_lookup(embeddings, eos_time_slice)
pad_step_embedded = tf.nn.embedding_lookup(embeddings, pad_time_slice)
#manually specifying loop function through time - to get initial cell state and input to RNN
#normally we'd just use dynamic_rnn, but lets get detailed here with raw_rnn

#we define and return these values, no operations occur here
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
        prediction = tf.argmax(output_logits, axis=1)
        # embed prediction for the next input
        next_input = tf.nn.embedding_lookup(embeddings, prediction)
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

#Creates an RNN specified by RNNCell cell and loop function loop_fn.
#This function is a more primitive version of dynamic_rnn that provides more direct access to the
#inputs each iteration. It also provides more control over when to start and finish reading the sequence,
#and what to emit for the output.
#ta = tensor array
decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
decoder_outputs = decoder_outputs_ta.stack()
decoder_outputs
#to convert output to human readable prediction
#we will reshape output tensor

#Unpacks the given dimension of a rank-R tensor into rank-(R-1) tensors.
#reduces dimensionality
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
sess.run(tf.global_variables_initializer())
batch_size = 100

def next_feed(start):
    batch = h.createBatch(sentences, start, batch_size)

    encoder_inputs_, encoder_input_lengths_ = helpers.batch(batch)
    decoder_targets_, _ = helpers.batch(
        [(sequence) + [EOS] + [PAD] * 2 for sequence in batch]
    )
    return {
        encoder_inputs: encoder_inputs_,
        encoder_inputs_length: encoder_input_lengths_,
        decoder_targets: decoder_targets_,
    }
loss_track = []
max_batches = int(len(sentences)/batch_size)
batches_in_epoch = 100

iteration=100
flag=0
try:
    while True:
        start=0
        for batch in range(max_batches):
            if start> len(sentences):
                break

            fd = next_feed(start)

            _, l = sess.run([train_op, loss], fd)
            loss_track.append(l)

            if batch == 0 or batch % batches_in_epoch == 0:
                mLoss= sess.run(loss,fd)
                if mLoss < 2:
                    flag=1
                    break
                print('batch {}'.format(batch))
                print('  minibatch loss: {}'.format(mLoss))
                predict_ = sess.run(decoder_prediction, fd)
                for i, (inp, pred) in enumerate(zip(fd[encoder_inputs].T, predict_.T)):
                    print('  sample {}:'.format(i + 1))
                    print('    input     > {}'.format(h.toWords(inp,reverseLookUp)))
                    print('    predicted > {}'.format(h.toWords(pred,reverseLookUp)))
                    print('    BLEUScore > '+str(nltk.translate.bleu_score.sentence_bleu([h.toWords(inp,reverseLookUp)], h.toWords(pred,reverseLookUp))))
                    if i >= 2:
                        break
                print()
            start=start+batch_size
        if flag==1:
            break

except KeyboardInterrupt:
    print('training interrupted')