#####################################
########## Anurag Banerjee ##########
########### 2018CSM1007 #############
#####################################	
import numpy as np
import itertools
import time
import math
import json
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy
import seaborn
import pandas


def gettimespent(sec):
    min = None
    if sec <= 60:
        return str(int(sec)) + str(' seconds')
    elif sec > 60:
        min = sec / 60
        sec = min - math.floor(min)
        min = int(min - sec)
        sec = int(sec * 60)
    if min <= 60:
        return str(min) + str(' minutes ') + str(sec) + str(' seconds')
    elif min > 60:
        hour = min / 60
        min = hour - math.floor(hour)
        hour = int(hour - min)
        min = int(min * 60)
        return str(hour) + str(' hours ') + str(min) + str(' minutes ') + str(sec) + str(' seconds')


def rounddwn(x):
    """
    Find nearest lower 100's number for given number
    :param x:
    :return:
    """
    return int(math.floor(x / 100.0)) * 100


def prepData(path, xdata, ydata, xfile, yfile, dictfile, flag):
    print("\nPreparing data for LSTM input...")
    start = time.time()     # begin timer

    blocksize = 4       # each image has 4 parts
    all_permut = list(itertools.permutations([i for i in range(1, blocksize+1)]))
    dim1 = int(xdata.shape[0]/blocksize)
    num_of_permut = 24
    X = np.zeros((num_of_permut*dim1, blocksize, xdata.shape[1]), dtype=np.float16)
    Y = np.zeros((num_of_permut*dim1, len(all_permut)), dtype=np.uint8)          # for one-hot encoding
    for_one_hot = np.eye(len(all_permut), dtype=np.uint8)       # identity matrix
    
    c = 1
    i = 0
    j = 0
    start_perm = time.time()
    while i < xdata.shape[0]:
        image = xdata[i:i+blocksize, :]        # 4 = blocksize parts form one image
        labels = ydata[i:i+blocksize, :]
        # creating class distributions
        #np.random.shuffle(all_permut)
        all_permut_part = all_permut[:num_of_permut]
        for idx, permute in enumerate(all_permut_part):
          xrow = np.empty((blocksize, xdata.shape[1]), dtype=np.float16)
          for k in range(blocksize):
            xrow[k,:] = image[permute[k]-1, :]
          X[j,:,:] = xrow
          Y[j,:] = for_one_hot[idx,:]
          j = j + 1
          del xrow
        # class distribution creation done

        if c % 500 == 0:
          end_perm = time.time()
          print("\t" + str(i) + " images processed... took "+gettimespent(end_perm-start_perm))
          start_perm = time.time()          

        i = i + blocksize
        c = i
    print("\nSaving Files...")
    #X = X[~np.all(X == 0, axis=1)]
    
    print(X.shape)
    print(Y.shape)
    c = 1
    if flag == 'train':
      limit = rounddwn(X.shape[0] * 1)
      i = 0
      while i < X.shape[0]:
        if i+limit > X.shape[0]:
          break
        print(path+xfile+str(c))
        np.save(path+xfile+str(c), X[i:i+limit,:,:])
        np.save(path+yfile+str(c), Y[i:i+limit,:])
        c = c + 1
        i = i + limit
      if i < X.shape[0]:
        np.save(path+xfile+str(c), X[i:,:,:])
        np.save(path+yfile+str(c), Y[i:,:])
      else:
        c = c - 1
    elif flag == 'test':
      np.save(path+xfile, X)
      np.save(path+yfile, Y)
    
    info = {"feat_in_img_part": xdata.shape[1], "num_of_img": X.shape[0], "blocksize": blocksize, "num_classes": len(all_permut), "num_of_files":c}
    with open(path+dictfile, 'w') as fout:
        json.dump(info, fout, indent=4)
    print("\nFiles saved...")
    end = time.time()
    del X
    del Y
    print("\nProcessing complete... took "+gettimespent(end-start))


def extract_batch_size(_train, step, batch_size):
    # Function to fetch a "batch_size" amount of data from "(X|y)_train" data.

    shape = list(_train.shape)
    shape[0] = batch_size
    batch_s = np.empty(shape)

    for i in range(batch_size):
        # Loop index
        index = ((step - 1) * batch_size + i) % len(_train)
        batch_s[i] = _train[index]

    return batch_s


def RNN_LSTM(_X, n_input, n_steps, n_hidden, _weights, _biases):
    # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, n_input])
    # new shape: (n_steps*batch_size, n_input)

    # ReLU activation
    _X = tf.nn.relu(tf.matmul(_X, _weights['hidden']) + _biases['hidden'])

    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(_X, n_steps, 0)
    # new shape: n_steps * (batch_size, n_hidden)

    # Define three stacked LSTM cells (three recurrent layers deep) with tensorflow
    lstm_cell_1 = tf.contrib.rnn.LSTMCell(n_hidden, name='basic_lstm_cell', forget_bias=1.4, state_is_tuple=True)
    lstm_cell_2 = tf.contrib.rnn.LSTMCell(n_hidden, name='basic_lstm_cell', forget_bias=2.0, state_is_tuple=True)
    lstm_cell_3 = tf.contrib.rnn.LSTMCell(n_hidden, name='basic_lstm_cell', forget_bias=1.9, state_is_tuple=True)
    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2, lstm_cell_3], state_is_tuple=True)
    # Get LSTM cell output
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)

    # Get last time step's output feature for a "many-to-one" style classifier,
    # as in the image describing RNNs at the top of this page
    lstm_last_output = outputs[-1]

    # Linear activation
    return tf.matmul(lstm_last_output, _weights['out']) + _biases['out']


def trainlstm(basepath, basextrain, baseytrain, trainfiles, xtest, ytest, param):
    
    tf.reset_default_graph()
    n_input, n_steps, n_hidden, n_classes, lambda_loss_amount, learning_rate, batch_size, \
                                                                        training_iters, display_iter = param
    # Graph input/output
    x = tf.placeholder(tf.float32, [None, n_steps, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])

    # Graph weights
    weights = {
        'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])),  # Hidden layer weights
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0))
    }
    biases = {
        'hidden': tf.Variable(tf.random_normal([n_hidden])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    pred = RNN_LSTM(x, n_input, n_steps, n_hidden, weights, biases)
    # print("\npred = "+str(pred))

    # Loss, optimizer and evaluation
    l2 = lambda_loss_amount * sum(
        tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
    )  # L2 loss prevents this overkill neural network to overfit the data

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)) + l2  # Softmax loss
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  # Adam Optimizer

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # print("\ncorrect_pred = "+str(correct_pred))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # To keep track of training's performance
    test_losses = []
    test_accuracies = []
    train_losses = []
    train_accuracies = []

    # Launch the graph
    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
    init = tf.global_variables_initializer()
    sess.run(init)

    # Perform Training steps with "batch_size" amount of example data at each loop
    
    # get initial train and test file
    # basepath, basextrain, baseytrain, trainfiles
    c = 1
    xtrain = np.load(basepath+basextrain+str(c)+".npy")
    ytrain = np.load(basepath+baseytrain+str(c)+".npy")
    # get initial train and test file complete
    
    step = 1
    while step * batch_size <= training_iters:
      batch_xs = extract_batch_size(xtrain, step, batch_size)
      batch_ys = extract_batch_size(ytrain, step, batch_size)

      # Fit training using batch data
      _, loss, acc = sess.run(
          [optimizer, cost, accuracy],
          feed_dict={
              x: batch_xs,
              y: batch_ys
          }
      )
      train_losses.append(loss)
      train_accuracies.append(acc)

      # Evaluate network only at some steps for faster training:
      if (step * batch_size % display_iter == 0) or (step == 1) or (step * batch_size > training_iters):
          # To not spam console, show training accuracy/loss in this "if"
          print("Training iter #" + str(step * batch_size) + \
                ":   Batch Loss = " + "{:.6f}".format(loss) + \
                ", Accuracy = {}".format(acc))

          # Evaluation on the test set (no learning made here - just evaluation for diagnosis)
          loss, acc = sess.run(
              [cost, accuracy],
              feed_dict={
                  x: xtest,
                  y: ytest
              }
          )
          test_losses.append(loss)
          test_accuracies.append(acc)
          print("PERFORMANCE ON TEST SET: " + \
                "Batch Loss = {}".format(loss) + \
                ", Accuracy = {}".format(acc))
          # Change training data set
#           if ((step * batch_size) == 0.5*training_iters):
#             # basepath, basextrain, baseytrain, trainfiles
#             c = (c + 1) % trainfiles
#             if c == 0:
#               c = c + 1
#             xtrain = np.load(basepath+basextrain+str(c)+".npy")
#             ytrain = np.load(basepath+baseytrain+str(c)+".npy")
#             print("Dataset changed...at "+str(step)+" * "+str(batch_size))
          # Changing train dataset complete
      
      step += 1

    print("Optimization Finished!")

    # Accuracy for test data

    one_hot_predictions, accuracy, final_loss = sess.run(
        [pred, accuracy, cost],
        feed_dict={
            x: xtest,
            y: ytest
        }
    )

    test_losses.append(final_loss)
    test_accuracies.append(accuracy)

    print("FINAL RESULT: " + \
          "Batch Loss = {}".format(final_loss) + \
          ", Accuracy = {}".format(accuracy))
    
    print("Testing Accuracy: {}%".format(round(100*accuracy),3))
    # Now plotting
    
    indep_train_axis = np.array(range(batch_size, (len(train_losses) + 1) * batch_size, batch_size))
    #print("train axis length "+str(len(indep_train_axis))+" accu length "+str(len(np.array(train_accuracies))))
    retain = 100
    xin = indep_train_axis[::retain]
    yin = np.array(train_accuracies[::retain])
    #print("train xin length "+str(len(xin))+" yin length "+str(len(yin)))
    plt.plot(xin, yin, "b--", label="Train accuracies")
    
    indep_test_axis = np.append(
        np.array(range(batch_size, len(test_losses) * display_iter, display_iter)[:-1]),
        [training_iters]
    )
    #print("train axis length "+str(len(indep_test_axis))+" accu length "+str(len(np.array(test_accuracies))))
    xin = indep_test_axis
    yin = np.array(test_accuracies)
    #print("test xin length "+str(len(xin))+" yin length "+str(len(yin)))
    plt.plot(xin, yin, "g-", label="Test accuracies")
    
    plt.title("Training session's progress over iterations")
    plt.legend(loc='upper left', shadow=True)
    plt.ylabel('Training Progress (Accuracy values)')
    plt.xlabel('Training iteration')

    plt.show()
    
    
    # Confusion Matrix
    predictions = one_hot_predictions.argmax(1)
    actuals = ytest.argmax(1)
    cm = confusion_matrix(actuals,predictions)
    dataframe_cm = pandas.DataFrame(cm,range(n_classes),range(n_classes))
    fig = plt.figure()
    plt.subplot(1, 1, 1)
    seaborn.heatmap(dataframe_cm,fmt='g', annot=True,annot_kws={"size": 8})
    plt.ylabel('Predicted Label')
    plt.xlabel('True Label')
    plt.title('Confusion Matrix after Complete Iterations')
    plt.tight_layout()
    plt.show()


def main():
    while True:
        print("\n-----------------Menu--------------")
        print("Enter 1 to prepare data for lstm input (takes around 5-10 minutes on i7, 8GB)")
        print("Enter 2 to perform LSTM operation")
        ch = int(input("Enter 0 to exit: "))

        if ch == 1:
            data = np.load('/content/gdrive/My Drive/data2411/datasubhro.npy')
            retain = data.shape[0]
            xdata = data[:retain, :-1]
            #print(xdata.shape)
            ydata = data[:retain, -1].reshape((xdata.shape[0], 1))
            del data
            # ------------Splitting train and test data----------
            split_percent = 0.8
            train_instances = rounddwn(xdata.shape[0] * split_percent)
            xtrain = xdata[:train_instances, :]
            xtest = xdata[train_instances:, :]

            ytrain = ydata[:train_instances, :]
            ytest = ydata[train_instances:, :]
            del xdata
            del ydata
            prepData('/content/gdrive/My Drive/data2411/', xtrain, ytrain, 'NlstmX_train', 'NlstmY_train', 'Ntraininfo.json', 'train')
            prepData('/content/gdrive/My Drive/data2411/', xtest, ytest, 'NlstmX_test', 'NlstmY_test', 'Ntestinfo.json', 'test')
            del xtrain
            del ytrain
            del xtest
            del ytest
        elif ch == 2:
            # -------------Read the files--------------
            with open('/content/gdrive/My Drive/data2411/Ntraininfo.json') as fin:
                traininfo = json.load(fin)

            #xtrain = np.load('/content/gdrive/My Drive/data2411/NlstmX_train.npy', datainfo['num_of_files'])
            #ytrain = np.load('/content/gdrive/My Drive/data2411/NlstmY_train.npy')
            
            xtest = np.load('/content/gdrive/My Drive/data2411/NlstmX_test.npy')
            ytest = np.load('/content/gdrive/My Drive/data2411/NlstmY_test.npy')

            # ---------------Input Data-----------------
            #training_data_count = len(xtrain)  # training series
            test_data_count = len(xtest)  # testing series
            n_steps = traininfo["blocksize"]  # timesteps per series
            n_input = traininfo["feat_in_img_part"]  # input parameters per timestep

            # ----LSTM Neural Network's internal structure-----
            n_hidden = 256  # Hidden layer num of features
            n_classes = traininfo["num_classes"]  # Total classes

            # ---------------Training---------------------
            learning_rate = 0.000448
            lambda_loss_amount = 0.00029
            training_iters = traininfo['num_of_img'] * 16  # Loop 100 times on the dataset
            print("\nTraining will run for "+str(training_iters)+" iterations...")
            batch_size = 128
            display_iter = 10000  # To show test set accuracy during training

            param = (n_input, n_steps, n_hidden, n_classes, lambda_loss_amount, learning_rate, batch_size,
                                                                                training_iters, display_iter)

            trainlstm('/content/gdrive/My Drive/data2411/', 'NlstmX_train', 'NlstmY_train',
                      traininfo['num_of_files'], xtest, ytest, param)
#             del xtrain
#             del ytrain
#             del xtest
#             del ytest

        elif ch == 0:
            break
        else:
            print("\nWrong Choice! Try Again.")
        
        
if __name__ == "__main__":
	main()
