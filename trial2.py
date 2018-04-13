import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.python.ops.rnn_cell import  LSTMStateTuple
from util import init_weight, get_ptb_data, display_tree

tf.reset_default_graph() #Clears the default graph stack and resets the global default graph.
sess = tf.InteractiveSession() #initializes a tensorflow session
tf.__version__


import nltk
import numpy as np


# TREEEEEEE
class Node(object):
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.count = 1


def insert(root, value):
    if not root:
        return Node(value)
    elif root.value == value:
        root.count += 1
    elif value < root.value:
        root.left = insert(root.left, value)
    else:
        root.right = insert(root.right, value)

    return root

def create(seq):
    root = None
    for word in seq:
        root = insert(root, word)

    return root



def print_tree(root):
    if root:
        print_tree(root.left)
        print(root.value)
        print_tree(root.right)


# from phase 1 divyas helpers
def getLookUps (path):
    #input: path of dataset
    #output: two lookups- 1 gives the index corresponding to each word in the vocabulry
    #and other which gives the word corresponding to the index
    file_content = open(path).read()
    sentences = nltk.tokenize.sent_tokenize(file_content)
    vocab = {}
    tokens = {}

    for a in sentences:
        tokens = nltk.word_tokenize(a)
        #del tokens[0]
        index=0
        for k in tokens:
            vocab[k] = 1
    vocab['EOS']=1
    vocab['PAD']=0
    index=2
    for key, value in vocab.items():
        vocab[key]=index
        index+=1
    reverseLookUp ={}

    for key, value in vocab.items():
        reverseLookUp[value]=key

    return vocab, reverseLookUp

def wordEmbeddings(sentence, lookUp):
    words = nltk.word_tokenize(sentence)
    embeddings =[]
    for w in words:
        embeddings.append(lookUp[w])
    return embeddings



def getSentences(path,vocab):
    file_content = open(path).read()
    sentences = nltk.tokenize.sent_tokenize(file_content)
    tokens = {}
    tokenList= []
    for a in sentences:
        tokens = nltk.word_tokenize(a)
        b= []
        #del tokens[0]
        for a in tokens :
            b.append(vocab[a])

        tokenList.append(b)
    return tokenList

def toWords(input, reverseLookUp):
    wordSequence =[]
    for element in input:
        if element in reverseLookUp:
            wordSequence.append(reverseLookUp[element])
        else:
            wordSequence.append('.')
    return wordSequence

datasetPath = "trees/treeSentences.txt"
lookUp, reverseLookUp = getLookUps(datasetPath)

sentences=getSentences(datasetPath,lookUp)
#print(lookUp)
#print(reverseLookUp)
#print(sentences)
print("hello")
vocabsize=len(lookUp)
#print("vocab size: "+str(len(lookUp))+" number of sentences: "+str(len(sentences)))

# word vector dimensions
dime=200


# word embedding

# all variables will get changed during back propogation

embeddings = tf.Variable(tf.random_uniform([vocabsize, dime], -1.0, 1.0), dtype=tf.float32)
W1 = tf.Variable(tf.random_uniform([dime, dime], -1, 1), dtype=tf.float32)
W2 = tf.Variable(tf.random_uniform([dime, dime], -1, 1), dtype=tf.float32)

# you will get the sentence embedding - modified yatharths
def SentenceEmbedding(tree):
    if tree.word is not None:
        # this is a leaf node
        x = tf.nn.embedding_lookup(embeddings, [tree.word])

    else:
        # this node has children
        x1 = SentenceEmbedding(tree.left)
        x2 = SentenceEmbedding(tree.right)
        act= tf.matmul(x1, W1) +  tf.matmul(x2, W2)
        x = tf.nn.relu( act)

    return x

#preparing the full input for decoder( we are not providing encoder final state as c,h. we are provding it as input(xs) to decoder)
def PrepareInput(sentence,tree_sent) :
    k=len(sentence)
    # treeof=create(sentence)
    print("-----------------------------")
    print(sentence)
    print(tree_sent)
    tup=SentenceEmbedding(tree_sent)
    tup=tup.eval()
    fullinput=[]
    for i in range(k):
        fullinput.append(tup)
    return fullinput

#from phase 1 same
def createBatch(sentences, startIndex, batchSize):
    if startIndex+batchSize<len(sentences):
        return sentences[startIndex:startIndex+batchSize]
    else:
        return sentences[startIndex:]
# encoding done

#from helper
#this function only returns a 12*10 matrix. 1 batch. we need to use the embeddings ansd reshape
def batchfun(inputs, max_sequence_length=None):
    """
    Args:
        inputs:
            list of sentences (integer lists)
        max_sequence_length:
            integer specifying how large should `max_time` dimension be.
            If None, maximum sequence length would be used

    Outputs:
        inputs_time_major:
            input sentences transformed into time-major matrix
            (shape [max_time, batch_size]) padded with 0s
        sequence_lengths:
            batch-sized list of integers specifying amount of active
            time steps in each input sequence
    """

    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)

    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)

    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32)  # == PAD change this to change pad

    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            inputs_batch_major[i, j] = element

    # [batch_size, max_time] -> [max_time, batch_size]
    inputs_time_major = inputs_batch_major.swapaxes(0, 1)

    return inputs_time_major, max_sequence_length

"""
def PrepareOutput(sentence):
    k = len(sentence)
    fulloutput = []
    for m in sentence :
        emb = tf.nn.embedding_lookup(embeddings, [m])
        fulloutput.append[emb]


    return fulloutput
"""

# data set max lenght = 12
# vocab size = 58

# REMOVED ATTENTION
# cleaned up the code

PAD = 0
EOS = 1


input_embedding_size = dime


decoder_hidden_units = 30

encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
# simply there, for printing output

decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
# same sentence

decoder_inputs = tf.placeholder(shape=(None, None,dime), dtype=tf.float32, name='decoder_inputs')
# sentence embedding multiple times

decoder_cell = tf.contrib.rnn.LSTMCell(decoder_hidden_units)


"""
efc = tf.Variable(tf.zeros([30]), validate_shape=False,dtype=tf.float32)
efh = tf.Variable(tf.zeros([30]),validate_shape=False,dtype=tf.float32 )

encoder_final_state = LSTMStateTuple(
    c=efc,
    h=efh
)
"""
efc = tf.placeholder(shape=(None, None), dtype=tf.int32, name='efc')
# simply there, for printing output

efh = tf.placeholder(shape=(None, None), dtype=tf.int32, name='efh')

encoder_final_state = LSTMStateTuple(
    c=efc,
    h=efh)

# WE COULD DO EITHER of these things

# provide initial state to lstm ( final encoder state)
# or provide our embedding as input to all the lstm decoder states- which i feel is better
# LOOK at sequence labelling like Part of speech tagging using lstm
# it is also variable lentgh and classification at every state. our number of classes is vocab size

# used dyanmic rnn istead of raw rnn- no loop function
decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(decoder_cell, decoder_inputs,dtype=tf.float32)

# check time major

print("i made it")

decoder_logits = tf.contrib.layers.fully_connected(decoder_outputs, vocabsize,activation_fn=tf.nn.relu)

decoder_prediction = tf.argmax(decoder_logits, 2)


stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(decoder_targets, depth=vocabsize, dtype=tf.int32),
    logits=decoder_logits,
)

loss = tf.reduce_mean(stepwise_cross_entropy)
train_op = tf.train.AdamOptimizer().minimize(loss)

sess.run(tf.global_variables_initializer())


batch_size = 10


def next_feed(start):
    batch = createBatch(sentences, start, batch_size)
    encoder_inputs_, encoder_input_lengths_ = batchfun(batch)
    decoder_targets_, _ = batchfun( [(sequence)  for sequence in batch])
    #correct till here
    # check this part
    ronaldo, q = batchfun(batch)
    #print(ronaldo)
    tenin = []
    train, test, word2idx = get_ptb_data()
    # print(len(np.transpose(ronaldo)))

    for k in range(0,len(np.transpose(ronaldo))):
        #k=PrepareInput(k))
        # print(k)
        tenin.append(PrepareInput(ronaldo[k],train[k]))
    #print(tf.shape(tenin))
    # we need to reshape this properly
    # [ batch size, sequence lenght,wordvec size
    #tenin.eval()
    #print(tenin)
    #decoder_inputs_ = tf.reshape(tenin, [10, 12, dime])
    tenin=np.reshape(tenin,(encoder_input_lengths_,batch_size,200))
    decoder_inputs_= tenin

    # this part causes the problem
    return {
        encoder_inputs: encoder_inputs_,
        decoder_inputs: decoder_inputs_,
        decoder_targets: decoder_targets_,
    }


# same as phase 1 code
loss_track = []
max_batches = int(len(sentences)/batch_size)
batches_in_epoch = 10

iteration=100
flag=0
try:
    while True:
        start=0
        writer = tf.summary.FileWriter('logs', graph=tf.get_default_graph())

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
                    print('    input     > {}'.format(toWords(inp,reverseLookUp)))
                    print('    predicted > {}'.format(toWords(pred,reverseLookUp)))
                    print('    BLEUScore > '+str(nltk.translate.bleu_score.sentence_bleu([toWords(inp,reverseLookUp)], toWords(pred,reverseLookUp))))
                    if i >= 2:
                        break
                print()
            start=start+batch_size
        if flag==1:
            writer.close()

            break

except KeyboardInterrupt:
    print('training interrupted')