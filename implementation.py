import tensorflow as tf
BATCH_SIZE = 128
MAX_WORDS_IN_REVIEW = 256  # Maximum length of a review to consider
EMBEDDING_SIZE = 50  # Dimensions for each word vector
numClasses=2
numberOfNeurons=150
# num_layers=2
punctuations = set('''!()-[]{};:'"\,<>./?@#$%^&*_~''')
stop_words = set({'ourselves', 'hers', 'between', 'yourself', 'again',
                  'there', 'about', 'once', 'during', 'out', 'very', 'having',
                  'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its',
                  'yours', 'such', 'into', 'of', 'most', 'itself', 'other',
                  'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him',
                  'each', 'the', 'themselves', 'below', 'are', 'we',
                  'these', 'your', 'his', 'through', 'don', 'me', 'were',
                  'her', 'more', 'himself', 'this', 'down', 'should', 'our',
                  'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had',
                  'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them',
                  'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does',
                  'yourselves', 'then', 'that', 'because', 'what', 'over',
                  'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you',
                  'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
                  'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being',
                  'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it',
                  'how', 'further', 'was', 'here', 'than'})
emotions=set([':(', ';)', ':|', ';(', ':)'])


def my_relu(in_value):
    """
    Implement a ReLU activation function that takes a scalar tf.placeholder as input
    and returns the appropriate output. For more information see the assignment spec.
    """
    const0=tf.constant(0,dtype=tf.float32)
    out_value=tf.maximum(in_value,const0)
    return out_value


def preprocess(review):
    """
    Apply preprocessing to a single review. You can do anything here that is manipulation
    at a string level, e.g.
        - removing stop words
        - stripping/adding punctuation
        - changing case
        - word find/replace
    RETURN: the preprocessed review in string form.
    """
    review = review.lower().replace("<br />", " ")
    review=review.split()
    processed_review=[]
    for word in review:
        if word in emotions:
            processed_review.append(word)
        else:
            new_word_list = []
            for char in word:
                if char not in punctuations:
                    new_word_list.append(char)
            new_word=''.join(new_word_list)
            if new_word not in stop_words:
                processed_review.append(new_word)
    return processed_review


def length(sequence):
  used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
  length = tf.reduce_sum(used, 1)+10
  length = tf.cast(tf.clip_by_value(tf.cast(length, tf.int32),0,MAX_WORDS_IN_REVIEW),tf.int32)
  return length



def define_graph():
    """
    Implement your model here. You will need to define placeholders, for the input and labels,
    Note that the input is not strings of words, but the strings after the embedding lookup
    has been applied (i.e. arrays of floats).

    In all cases this code will be called by an unaltered runner.py. You should read this
    file and ensure your code here is compatible.

    Consult the assignment specification for details of which parts of the TF API are
    permitted for use in this function.

    You must return, in the following order, the placeholders/tensors for;
    RETURNS: input, labels, optimizer, accuracy and loss
    # """
    tf.reset_default_graph()
    dropout_keep_prob=tf.placeholder_with_default(0.6,(),name="dropout_keep_prob")
    input_data=tf.placeholder(tf.float32,(BATCH_SIZE, MAX_WORDS_IN_REVIEW, EMBEDDING_SIZE),name="input_data")
    labels=tf.placeholder(tf.float32,(BATCH_SIZE,numClasses),name="labels")
    lstmCell = tf.contrib.rnn.BasicLSTMCell(numberOfNeurons)
    lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=dropout_keep_prob)
    Wh=tf.Variable(tf.truncated_normal([numberOfNeurons, numberOfNeurons]))
    da=int(numberOfNeurons)
    Wu=tf.Variable(tf.truncated_normal([da, da]))
    W=tf.Variable(tf.truncated_normal([numberOfNeurons+da,1]))
    Wp = tf.Variable(tf.truncated_normal([numberOfNeurons,numberOfNeurons]))
    Wx = tf.Variable(tf.truncated_normal([numberOfNeurons, numberOfNeurons]))
    Va = tf.Variable(tf.truncated_normal([1,da]))
    e=tf.fill([MAX_WORDS_IN_REVIEW,1],1.0)
    weight = tf.Variable(tf.truncated_normal([numberOfNeurons, numClasses]))
    bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))



    H, _ = tf.nn.dynamic_rnn(lstmCell, input_data, dtype=tf.float32)
    Wh_mulH=tf.tensordot(H,Wh,((2),(0)))
    Vea=tf.matmul(e,Va)
    Wu_mulVea=tf.matmul(Vea,Wu)
    Wu_mulVea_tile=tf.tile(Wu_mulVea,(BATCH_SIZE,1))
    Wu_mulVea_tile=tf.reshape(Wu_mulVea_tile,(-1,MAX_WORDS_IN_REVIEW,da))
    M=tf.concat((Wh_mulH,Wu_mulVea_tile),2)
    M=tf.tanh(M)
    beforeSoftmax=tf.tensordot(M,W,((2),(0)))
    beforeSoftmax=tf.squeeze(beforeSoftmax)
    alpha=tf.nn.softmax(beforeSoftmax)
    alpha=tf.expand_dims(alpha, 1)
    R=tf.matmul(alpha,H)
    R=tf.squeeze(R)
    Wp_mulR=tf.matmul(R,Wp)
    value = tf.transpose(H, [1, 0, 2])
    last = tf.gather(value, int(value.get_shape()[0]) - 1)
    Wx_mulLast=tf.matmul(last,Wx)
    h_star=tf.tanh(Wp_mulR+Wx_mulLast)
    prediction = (tf.matmul(h_star, weight) + bias)
    correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32),name="accuracy")
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=labels),name="loss")
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    return input_data, labels, dropout_keep_prob, optimizer, accuracy, loss
