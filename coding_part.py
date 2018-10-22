
# coding: utf-8

# <h3><center>Project topic:TensorFlow and Neural Networks Applications in HealthCareÿ</center></h3>
# 
# <h5><center>Student Name: Liangliang Zhang</center></h5>

# ### Problem Statement: 
# Use TensorFlow to build a Dense Neural Network that will be used to automatically classify fetal cardiotocogram to different fetal state (N, S, P) based on their diagnostic features data provided by the UCI Machine Learning Repository.

# ### Overview of Technology
# TensorFlow and TensorBoard was used to built Multilayer Dense Neural Network model and monitor loss function on training dataset.  
# 
# TensorFlow is an open source software library for numerical computation using data flow graphs. Nodes in the graph represent mathematical operations, while the graph edges represent the multidimensional data arrays (tensors) communicated between them. The flexible architecture allows you to deploy computation to one or more CPUs or GPUs in a desktop, server, or mobile device with a single API. (https://www.tensorflow.org/)

# ### Descreption of  Data
# 
# 2126 fetal cardiotocograms (CTGs) were automatically processed and the respective diagnostic features measured. The CTGs were also classified by three expert obstetricians and a consensus classification label assigned to each of them. Classification was both with respect to a morphologic pattern (A, B, C. ...) and to a fetal state (N=normal; S=suspect; P=pathologic).
# URL: http://archive.ics.uci.edu/ml/machine-learning-databases/00193/
# 
# Size: 1.66 MB., sample size: 2130
# 
# Format of data file: .xls file of Microsoft Excel

# ### Hardware
# 
# Windows PC with Intel Core M-5Y10c CPU (0.8GHz, 998MHz) and 4GB RAM

# ### Sofeware
# 
# Anaconda with Python 3.6.1
# 
# TensorFlow 1.3.0 https://pypi.python.org/pypi/tensorflow/1.3.0

# ### Lessons learned & Pros/Cons
# 
# After tuning, my final Neural Network model gives a prediction accuracy of ~92% in training data and a prediction accuracy of ~90% in validation data.  This model performs reasonably well and I suppose that if we have more observations, especially observations of the minority class, we could have built a more powerful neural network.

# ### YouTube URLs:

# In[1]:


# import libraries
import tensorflow as tf
import numpy as np
import pandas as pd


# ### Steps & Demonstration

# #### 1. load data
# 
# Data fiel CTG.xls was downloaded from http://archive.ics.uci.edu/ml/machine-learning-databases/00193/ 

# In[2]:


#load data
ctg = pd.read_excel('CTG.xls', sheetname = 2)


# #### 2. Data cleaning

# In[3]:


#check data shape
print ('CTG data shape:,', ctg.shape)
#check data head
ctg.head()


# In[4]:


#check data tail
ctg.tail()


# Drop the column of Filename, Date and SegFile, as these information has absolutely no predictive power for determing the state of a CTG image. Keeping these information will only confuse our model when training neural network.

# In[5]:


ctg.drop(['FileName', 'Date', 'SegFile'], axis = 1, inplace = True)


# Drop first row as it is blank, then drop a few rows from the bottom as they contain  meaningless information.

# In[6]:


ctg_clear = ctg.drop(ctg.index[[0, 2127, 2128, 2129]])


# Check if there are any missing values.

# In[7]:


print ('Having missing values? :', ctg_clear.isnull().any().any())


# Let's check the shape and statistical descriptions after cleaning:

# In[8]:


print ('Data shape after cleaning', ctg_clear.shape)


# In[33]:


ctg_clear.head()


# In[10]:


ctg_clear.describe()


# #### 3. Extract feature and lables
# 
# The dataset has two types of labels: morphologic pattern and fetal state. In this project, we only use fetal state label to perform a 3-class classification. Then fetal labels was then onehot encoded to dummy variables.

# In[34]:


features = ctg_clear.iloc[:, :-12].values
labels = ctg_clear.iloc[:, -1].values

labels_onehot = pd.get_dummies(labels)

print ('Number of abservations:', features.shape[0])
print ('Number of features:', features.shape[1])
print ('number of labels:', labels_onehot.shape[1])


# #### 4. Train and validation data split
# 
# Then entire dataset was randomly split into training (80% 1700 cases) and validation (20% 426 cases) dataset. Train dataset is used for training our neural network, and validation datasetis used for testing the accurarcy of our model. 

# In[184]:


from sklearn.model_selection import train_test_split
# Take 1/5 images from the training data, and leave the remainder in training
train_dataset, valid_dataset, train_labels, valid_labels = train_test_split(features, labels_onehot.values, test_size=0.2, random_state=38)
print('Training data/label shape: ', train_dataset.shape, train_labels.shape)
print('Validation data/label shape: ', valid_dataset.shape, valid_labels.shape)


# In[185]:


#check the propotion of each class in train and validation data
print ('Propotion for each class in train data:', np.sum(train_labels, axis=0)/train_labels.shape[0])
print ('Propotion for each class in validaion data:', np.sum(valid_labels, axis=0)/valid_labels.shape[0])


# The propotion for each class is similar in training and validation dataset, so we will have all information needed in training data.

# #### 5. Dense Neural Network (DNN) model

# #### 5.1 Define a few useful  functions

# In[186]:


# calculate accuracy by identifying validation cases where the model's highest-probability class matches the true y label 
def accuracy(predictions, labels):
    correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
    accuracy_pct = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) * 100.0
    #another way to calculate this is to use np like following
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])
    #return accuracy_pct.eval()


# In[187]:


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=1e-4)
    #initial = tf.truncated_normal(shape, stddev=np.sqrt(2.0/shape[0]))
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    #initial = tf.constant(0.1, shape=shape)
    initial = tf.zeros(shape)
    return tf.Variable(initial, name)

split_by_half = lambda x,k : int(x/2**k)


# #### 5.2 Simple 2-layer DNN model with GradientDescentOptimizer

# In[188]:


valid_dataset = valid_dataset.astype(np.float32)
n_labels = 3
batch_size = 99
flattened_size = train_dataset.shape[1]
hidden_nodes = 100

graph = tf.Graph()
with graph.as_default():

    # Input data.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, flattened_size), name="TrainingData")
    tf_train_labelset = tf.placeholder(tf.float32, shape=(batch_size, n_labels), name="TrainingLabels")
    tf_valid_dataset = tf.constant(valid_dataset, name="ValidationData")
    
    # Variables.
    layer1_weights = tf.Variable(tf.truncated_normal([flattened_size, hidden_nodes]), name="weights1")
    layer1_biases = tf.Variable(tf.zeros([hidden_nodes]), name="biases1")
    layer2_weights = tf.Variable(tf.truncated_normal([hidden_nodes, n_labels]), name="weights2")
    layer2_biases = tf.Variable(tf.ones([n_labels]), name="biases2")

    # Model.
    def model(data, name):
        with tf.name_scope(name) as scope:
            layer1 = tf.add(tf.matmul(data, layer1_weights), layer1_biases, name="layer1")
            hidden1 = tf.nn.relu(layer1, name="relu1")
            layer2 = tf.add(tf.matmul(hidden1, layer2_weights), layer2_biases, name="layer2")
            return layer2 
    
    # Training computation.
    logits = model(tf_train_dataset, name="logits")
    #loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels = tf_train_labelset), name="loss")

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

    # Predictions for the training, validation
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset, name="validation"))


# In[189]:


# define run model function
def run_session(num_epochs, name):
    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run() 
        merged = tf.summary.merge_all()  
        writer = tf.summary.FileWriter("tmp/tensorflowlogs", session.graph)
        print("Initialized model:", name)
        for epoch in range(num_epochs):
            offset = (epoch * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            feed_dict = {tf_train_dataset : batch_data, tf_train_labelset : batch_labels}
            _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
            if (epoch % 500 == 0):
                print('Minibatch loss at epoch %d: %f' % (epoch, l))
                print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
                print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))


# In[179]:


run_session(5001, "DNN_2layer")


# After 5000 epoches, both training and validation accuracies are arround 76%, which is similar to a blind guess of first class.
# 
# Next, we modify several parameters of our DNN model to see if we can improve model performance. Modifications are listed below:
# 1.	More hidden layers
# 2.	Regularization and dropout to avoid over fitting
# 3.	Altinative optimizer
# 
# Also, summary for loss function, train accuracy and validation accuracy were added to TensroBoard, so we can keep tracking our model performance.

# #### 5.3 4-layer DNN model with regularization, dropout and  AdamOptimizer

# In[194]:


batch_size = 340
flattened_size = train_dataset.shape[1]
hidden_nodes = 512
lamb_reg = 0.001
learning_rate = 0.001  #  learning rate for the momentum optimizer

graph = tf.Graph()
with graph.as_default():

    # Input data.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, flattened_size), name="TrainingData")
    tf_train_labelset = tf.placeholder(tf.float32, shape=(batch_size, n_labels), name="TrainingLabels")
    tf_valid_dataset = tf.constant(valid_dataset, name="ValidationData")
    tf_valid_labelset = tf.constant(valid_labels, name="ValidationLabels")
    # Variables.
    layer1_weights = weight_variable([flattened_size, hidden_nodes], name="weights1")
    layer1_biases = bias_variable([hidden_nodes], name="biases1")
    layer2_weights = weight_variable([hidden_nodes, split_by_half(hidden_nodes,1)], name="weights2")
    layer2_biases = bias_variable([split_by_half(hidden_nodes,1)], name="biases2")
    layer3_weights = weight_variable([split_by_half(hidden_nodes,1), split_by_half(hidden_nodes,2)], name="weights3")
    layer3_biases = bias_variable([split_by_half(hidden_nodes,2)], name="biases3")
    layer4_weights = weight_variable([split_by_half(hidden_nodes,2), n_labels], name="weights4")
    layer4_biases = bias_variable([n_labels], name="biases4")
        
    keep_prob = tf.placeholder("float", name="keep_prob")
    
    def model(data, name, proba=keep_prob):
        with tf.name_scope(name) as scope:
            layer1 = tf.add(tf.matmul(data, layer1_weights), layer1_biases, name="layer1")
            hidden1 = tf.nn.dropout(tf.nn.relu(layer1), proba, name="dropout1")   # dropout on the hidden layer
            layer2 = tf.add(tf.matmul(hidden1, layer2_weights), layer2_biases, name="layer2")  # a new hidden layer
            hidden2 = tf.nn.dropout(tf.nn.relu(layer2), proba, name="dropout2")
            layer3 = tf.add(tf.matmul(hidden2, layer3_weights), layer3_biases, name="layer3")
            hidden3 = tf.nn.dropout(tf.nn.relu(layer3), proba)
            layer4 = tf.add(tf.matmul(hidden3, layer4_weights), layer4_biases, name="layer4")
            return layer4
    
    # Training computation.
    logits = model(tf_train_dataset, "logits", keep_prob)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels = tf_train_labelset), name="loss")
    regularizers = (tf.nn.l2_loss(layer1_weights) + tf.nn.l2_loss(layer1_biases) +
                    tf.nn.l2_loss(layer2_weights) + tf.nn.l2_loss(layer2_biases) +
                    tf.nn.l2_loss(layer3_weights) + tf.nn.l2_loss(layer3_biases) +
                    tf.nn.l2_loss(layer4_weights) + tf.nn.l2_loss(layer4_biases) )

    # Add the regularization term to the loss.
    loss += lamb_reg * regularizers
    #loss = tf.reduce_mean(loss + lamb_reg * regularizers)
    
    # Optimizer
    #global_step = tf.Variable(0, name="globalstep")  # count  number of steps taken.
    #optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(loss, global_step=global_step)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001, epsilon=1e-04).minimize(loss)
    
    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset, "validation", 1.0))  # no dropout
    #saver = tf.train.Saver()   # a saver variable to save the model
    
    # acuuracy for training data
    train_correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.float32), (tf.cast(tf.argmax(tf_train_labelset, 1), tf.float32)))
    accuracy_train = tf.reduce_mean(tf.cast(train_correct_prediction, tf.float32))
    # acuuracy for validation data
    valid_correct_prediction = tf.equal(tf.cast(tf.argmax(model(tf_valid_dataset, "validation", 1.0), 1), tf.float32), (tf.cast(tf.argmax(tf_valid_labelset, 1), tf.float32)))
    accuracy_valid = tf.reduce_mean(tf.cast(valid_correct_prediction, tf.float32))   


# In[195]:


def run_session_2(num_epochs, name, k_prob=1.0):

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run() 
        
        # summaries
        loss_summary = tf.summary.scalar('Loss', loss)
        
        train_accuracy_summary = tf.summary.scalar('train_accuracy', accuracy_train)
        valid_accuracy_summary = tf.summary.scalar('valid_accuracy', accuracy_valid)
        
        merged = tf.summary.merge_all()  
        writer = tf.summary.FileWriter("tmp/tensorflowlogs_3", session.graph)
        
        print('Initialized model:', name,"\n")
        for epoch in range(num_epochs):
            offset = (epoch * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            feed_dict = {tf_train_dataset : batch_data, tf_train_labelset : batch_labels, keep_prob : k_prob}
            _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
            writer.add_summary(loss_summary.eval(feed_dict=feed_dict), epoch)
            writer.add_summary(train_accuracy_summary.eval(feed_dict=feed_dict), epoch)
            writer.add_summary(valid_accuracy_summary.eval(feed_dict=feed_dict), epoch)
            #writer.add_summary(learning_rate_summary.eval(), epoch)
            if (epoch % 500 == 0):
                print("Minibatch loss at epoch {}: {}".format(epoch, l))
                print("Minibatch accuracy: {:.1f}".format(accuracy(predictions, batch_labels)))
                print("Validation accuracy: {:.1f}\n".format(accuracy(valid_prediction.eval(), valid_labels)))
        #save_path = saver.save(session, "tmp/" + name +".ckpt")
        #print("Model saved in file: %s" % save_path)


# In[196]:


run_session_2(5001, "DNN_4layer_Adam", 1.0)


# In[145]:


run_session_2(5001, "DNN_4layer_Adam", 1.0)


# Modified Neural Network model gives a prediction accuracy of ~99%  in training data and a prediction accuracy of ~90%  in validation data. Visualization from TensroBoard is shown below:

# In[198]:


from IPython.display import Image
Image("TensorBoard.png")


# Different optimizer (MomentumOptimizer, AdamOptimizer, GradientDescentOptimizer), learning rate (0.0001, 0.001, 0.01, 0.1) and keep probability (1.0, 0.8, 0.5) were tested. The final DNN model (AdamOptimizer, learning rate =0.001, keep probability=1.0), which has the best performance on validation data, was shown above.

# #### 6. Conclusion
# 
# Our final Neural Network model gives a prediction accuracy of ~92% in training data and a prediction accuracy of ~90% in validation data. This model performs reasonably well and I suppose that if we have more observations, especially observations of the minority class, we could have built a more powerful neural network.
