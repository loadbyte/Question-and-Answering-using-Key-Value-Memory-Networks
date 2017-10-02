"""Description:- Webapp to run the Hashed key value network on facebook bAbI-task dataset.It takes a joint trained model on entire dataset and populates
a randomly selected story from the dataset to user and also populates the original question from dataset. The question can be changed according to story by 
user and depending on question it evaluates the model to generate output answer.It uses flask to run server"""


from __future__ import absolute_import
from __future__ import print_function

from data_utils import load_task, vectorize_data
from sklearn import cross_validation, metrics
from hashed_mem_nw import Hashed_Mem_Nw
from itertools import chain
from six.moves import range, reduce
from hashed_mem_nw import zero_nil_slot, add_gradient_noise
import time

import tensorflow as tf
import numpy as np
import pandas as pd
import random
import glob
import flask
import unicodedata
import warnings


app = flask.Flask(__name__)
timestamp = str(int(time.time()))
data,vocab,word_idx,sentence_size,memory_size,vocab_size=None,None,None,None,None,None#Global variables to store dictionary and data.
#index=1

# Defining initial parameters for the model which can be overwritten from command line.
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate for Adam Optimizer.")
tf.flags.DEFINE_float("epsilon", 0.1, "Epsilon value for Adam Optimizer.")
tf.flags.DEFINE_float("l2_lambda", 0.1, "Lambda for l2 loss.")
tf.flags.DEFINE_float("keep_prob", 1.0, "Keep probability for dropout")
tf.flags.DEFINE_float("max_grad_norm", 20.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("evaluation_interval", 20, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 50, "Batch size for training.")
tf.flags.DEFINE_integer("hops", 3, "Number of hops in the Memory Network.")
tf.flags.DEFINE_integer("epochs", 200, "Number of epochs to train for.")
tf.flags.DEFINE_integer("embedding_size", 40, "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("memory_size", 50, "Maximum size of memory.")
tf.flags.DEFINE_integer("random_state", None, "Random state.")
tf.flags.DEFINE_string("data_dir", "data/tasks_1-20_v1-2/en/", "Directory containing bAbI tasks")
tf.flags.DEFINE_string("param_output_file", "logs/params_{}.csv".format(timestamp), "Name of output file for model hyperparameters")
tf.flags.DEFINE_string("output_file", "logs/scores_{}.csv".format(timestamp), "Name of output file for final bAbI accuracy scores.")
tf.flags.DEFINE_integer("feature_size", 50, "Feature size")
tf.flags.DEFINE_string("reader", "bow", "Reader for the model")
FLAGS = tf.flags.FLAGS

#Below method processes data and makes a dictionary as well as reverse dictionary.It also initializes initial model.
def getdata():
    ids = range(1, 21)
    train, test = [], []
    for i in ids:
        tr, te = load_task(FLAGS.data_dir, i)
        train.append(tr)
        test.append(te)
    data = list(chain.from_iterable(train + test))
    vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in data)))
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

    max_story_size = max(map(len, (s for s, _, _ in data)))
    mean_story_size = int(np.mean([len(s) for s, _, _ in data]))
    sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in data)))
    query_size = max(map(len, (q for _, q, _ in data)))
    memory_size = min(FLAGS.memory_size, max_story_size)
    vocab_size = len(word_idx) + 1 # +1 for nil word
    sentence_size = max(query_size, sentence_size) # for the position

    tf.set_random_seed(FLAGS.random_state)
    batch_size = FLAGS.batch_size
    
        
    global_step = tf.Variable(0, name="global_step", trainable=False)
        
    starter_learning_rate = FLAGS.learning_rate
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 90000, 0.96, staircase=True)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=FLAGS.epsilon)

    model = Hashed_Mem_Nw(vocab_size=vocab_size,
                    query_size=sentence_size, story_size=sentence_size, memory_key_size=memory_size,
                     memory_value_size=memory_size, embedding_size=FLAGS.embedding_size, reader=FLAGS.reader, l2_lambda=FLAGS.l2_lambda)
    global model
    grads_and_vars = optimizer.compute_gradients(model.loss_op)
    grads_and_vars = [(tf.clip_by_norm(g, FLAGS.max_grad_norm), v)
                       for g, v in grads_and_vars if g is not None]
    grads_and_vars = [(add_gradient_noise(g), v) for g, v in grads_and_vars]
    nil_grads_and_vars = []
    for g, v in grads_and_vars:
        if v.name in model._nil_vars:
            nil_grads_and_vars.append((zero_nil_slot(g), v))
        else:
            nil_grads_and_vars.append((g, v))
    train_op = optimizer.apply_gradients(nil_grads_and_vars, name="train_op", global_step=global_step)
       

    return data,vocab,word_idx,sentence_size,memory_size,vocab_size

# Fetching a random story and related question from the dataset to be populated.
def story():
    random.seed(a=None)
    index=random.randint(1, len(data)-1)
    global index
    storyS=''
    for s,_,_ in data[index-1:index]:
        for sent in s:
            word1=''
            i=0
            for word in sent[:-1]:
                if(i==0):
                    word1=word1+word.title()+' '
                else:
                    word1=word1+word+' '
                i=i+1
            word1=word1+sent[-1]+'.'
            
            storyS=storyS+word1+'\n'
    
    question=''
    i=0
    for _,q,_ in data[index-1:index]:
        for word in q[:-1]:
            if(i==00):
                question=question+word.title()+' '
            else:
                question=question+word+' '
            i=i+1
        question=question+q[-1]+'?'
    
    return storyS,index,question

# It returns the vector form of story populated to user.
def getVecor(index):
    
    start=int(index)-1
    end=int(index)
    
    sV,Q,_ = vectorize_data(data[start:end], word_idx, sentence_size, memory_size)
    return sV,Q

# Below method vectorize the question sent by the user.
def vectorize_question(question):
    quest=[]
    quest1=np.zeros(sentence_size)
    
    question = question.strip()
    if question[-1] == '?':
        question = question[:-1]
    qwords = question.rstrip().lower().split() 
    i=0
    for word in qwords:
        
        quest1[i]=word_idx[word]
        i=i+1
    
    quest.append(quest1)
    return np.array(quest)

# Below method evaluates the story and question on pre trained model to generate output. 
def evalModel(sV,Q):
        
    FLAGS._parse_flags()
    
    with open(FLAGS.param_output_file, 'w') as f:
        for attr, value in sorted(FLAGS.__flags.items()):
            line = "{}={}".format(attr.upper(), value)
            f.write(line + '\n')
            

    print("Started Joint Model")

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, "./models/joint/model.ckpt") #Storing the pretrained model.
        def test_step(s, q):
            feed_dict = {
                model._query: q,
                model._memory_key: s,
                model._memory_value: s,
                model.keep_prob: 1
            }
            preds = sess.run(model.predict_op, feed_dict)  #Evaluating the model for given question and story.
            return preds
        
        test_preds = test_step(sV, Q)
        return vocab[test_preds[0]-1]

# Starting the server.
def run():
    app.run()

# Rendering the home page to browser.
@app.route('/')
def index():
    
    return flask.render_template("index.html")

# Populating the story when request for new story arrives from browser.
@app.route('/get/story', methods=['GET'])
def get_story():
    s,i,q=story()
    return flask.jsonify({
        "question" : q,
        "story": s,
        "question_idx": i
        
    })

# Passing the predicted answer to user.
@app.route('/get/answer', methods=['GET'])
def get_answer():
    question_idx  = flask.request.args.get('question_idx')
    user_question = flask.request.args.get('user_question')
    global user_question
    q=vectorize_question(user_question)
    
    sV,Q=getVecor(question_idx)
    ans=evalModel(sV,q)
    pred_answer=ans
    print(ans)
    if(len(ans)==0):
        pred_answer="Question can't be answered!!!"
    return flask.jsonify({
        "pred_answer" : pred_answer
        
    })


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    data,vocab,word_idx,sentence_size,memory_size,vocab_size=getdata()
    run()
   
    
