#Name- Subhranil Bagchi
#Entry Number- 2018CSY0002

import tensorflow as tf
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time
import numpy
import seaborn
import pandas
import itertools

def plot_graph(cnn_train,cnn_test,itr_interval):
    intervals = [x for x in range(1,len(cnn_train)*itr_interval+1,itr_interval)]
    plt.figure(1)
    plt.plot(intervals,cnn_train,color='red',label='Train')
    plt.plot(intervals,cnn_test,color='blue',label='Test')
    plt.xlabel("Iterations with Interval gap of 50")
    plt.ylabel("Accuracy in %")
    plt.title("Accuracy vs. No. of Iterations")
    plt.legend(loc=2)
    plt.show()

def new_conv_layer(input,num_input_channels,filter_size,num_filters,use_pooling=True):
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05))
    biases = tf.Variable(tf.constant(0.05, shape=[num_filters]))
    layer = tf.nn.conv2d(input=input,filter=weights,strides=[1, 1, 1, 1],padding='SAME')
    layer += biases
    if use_pooling:
        layer = tf.nn.max_pool(value=layer,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')
    layer = tf.nn.relu(layer)
    return layer

def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements() 
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features

def new_fc_layer(input,num_inputs,num_outputs,use_relu):
    shape=[num_inputs, num_outputs]
    weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05))
    biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]))
    layer = tf.matmul(input, weights) + biases
    if use_relu: layer = tf.nn.relu(layer)
    return layer

def getdata(op,data,data_size):
    itemsTaken = data
    imageTaken = numpy.reshape(itemsTaken,(data_size,img_size,img_size))
    x_LT = imageTaken[:,0:cut_size,0:cut_size]
    x_RT = imageTaken[:,0:cut_size,cut_size:img_size]
    x_LD = imageTaken[:,cut_size:img_size,0:cut_size]
    x_RD = imageTaken[:,cut_size:img_size,cut_size:img_size]
    y_true = numpy.zeros((data_size,num_classes))
    x_items = numpy.zeros((data_size,img_size,img_size))
    items_img = numpy.zeros((num_cut,cut_size,cut_size))
    #tempIndex = numpy.array([0,1,2,3])
    list_permute = numpy.array(list(itertools.permutations([0,1,2,3])))
    for indexImg in range(data_size):
        if indexImg % num_classes == 0: numpy.random.shuffle(list_permute)
        tempIndex = list_permute[(indexImg % num_classes)]
        #numpy.random.shuffle(tempIndex)
        items_img[tempIndex[0]] = x_LT[indexImg]
        items_img[tempIndex[1]] = x_RT[indexImg]
        items_img[tempIndex[2]] = x_LD[indexImg]
        items_img[tempIndex[3]] = x_RD[indexImg]
        x_items[indexImg,0:cut_size,0:cut_size] = items_img[0]
        x_items[indexImg,0:cut_size,cut_size:img_size] = items_img[1]
        x_items[indexImg,cut_size:img_size,0:cut_size] = items_img[2]
        x_items[indexImg,cut_size:img_size,cut_size:img_size] = items_img[3]
        for perIndx in range(len(op)):
            if tempIndex[0]==op[perIndx][0] and tempIndex[1]==op[perIndx][1] and tempIndex[2]==op[perIndx][2] and tempIndex[3]==op[perIndx][3]:
                y_true[indexImg,perIndx] = 1
                break;
    x_items = numpy.reshape(x_items,(data_size,img_size_flat))
    return(x_items,y_true)

def getbatch(j,batch_size,dataFeature,dataTarget):
    x_batch = dataFeature[j:j+batch_size]
    y_true_batch = dataTarget[j:j+batch_size]
    return(x_batch,y_true_batch)
    

def optimize(list_itr):
    start_time = time.time()
    graph_index = 0
    for i in range(num_iterations):
        if i % 3 == 0: dataFeature, dataTarget = getdata(outputPermute,data,data_size)
        j = 0
        count = 0
        while j<data_size:
            count += 1
            if train_batch_size<data_size-j: itr_batch_size = train_batch_size
            else: itr_batch_size = data_size-j
            x_batch, y_true_batch= getbatch(j,train_batch_size, dataFeature, dataTarget)
            feed_dict_train = {x: x_batch, y_true: y_true_batch}
            session.run(optimizer, feed_dict=feed_dict_train)
            if count % itr_interval == 0:
                acc1 = session.run(accuracy, feed_dict=feed_dict_train)
                k = 0
                correct_sum = 0
                cls_pred = numpy.zeros(data_test.shape[0], dtype=numpy.int)
                while k<data_test.shape[0]:
                    if test_batch_size<data_test.shape[0]-k: itr_test = test_batch_size
                    else: itr_test = data_test.shape[0]-k
                    test_batch, testTarget_batch = getbatch(k,test_batch_size, testFeature, testTarget)
                    feed_dict_test = {x: test_batch, y_true: testTarget_batch}
                    cls_pred[k:k+itr_test] = session.run(y_pred_cls, feed_dict=feed_dict_test)
                    actual_class = numpy.argmax(testTarget_batch, axis=1)
                    test_prediction = numpy.equal(cls_pred[k:k+itr_test], actual_class)
                    numpy.sum(test_prediction)
                    correct_sum = correct_sum + numpy.sum(test_prediction)
                    #cm = confusion_matrix(actual_class,cls_pred[k:k+itr_test])
                    k = k + itr_test
                acc2 = float(correct_sum)/data_test.shape[0]
                accuracyList_train[graph_index][0] = str(round(acc1*100,3))
                accuracyList_test[graph_index][0] = round(acc2*100,3)
                print("Epoch: " + str(i+1) +"\tIterations: " + str(count) + "\t\tTraining Accuracy: " + str(round(acc1*100,3)) + "%\tTesting Accuracy: " + str(round(acc2*100,3)) +"%")
                #print(graph_index,(list_itr)-1)
                graph_index += 1
            j = j + itr_batch_size

    k = 0
    correct_sum = 0
    confusion_sum = numpy.zeros((num_classes,num_classes))
    while k<data_test.shape[0]:
        if test_batch_size<data_test.shape[0]-k: itr_test = test_batch_size
        else: itr_test = data_test.shape[0]-k
        test_batch, testTarget_batch = getbatch(k,test_batch_size, testFeature, testTarget)
        feed_dict_test = {x: test_batch, y_true: testTarget_batch}
        cls_pred[k:k+itr_test] = session.run(y_pred_cls, feed_dict=feed_dict_test)
        actual_class = numpy.argmax(testTarget_batch, axis=1)
        test_prediction = numpy.equal(cls_pred[k:k+itr_test], actual_class)
        numpy.sum(test_prediction)
        correct_sum = correct_sum + numpy.sum(test_prediction)
        cm = confusion_matrix(actual_class,cls_pred[k:k+itr_test])
        confusion_sum = confusion_sum + cm
        k = k + itr_test
    acc = float(correct_sum)/data_test.shape[0]    
    dataframe_cm = pandas.DataFrame(confusion_sum,range(num_classes),range(num_classes))
    fig = plt.figure()
    plt.subplot(1, 1, 1)
    seaborn.heatmap(dataframe_cm,fmt='g', annot=True,annot_kws={"size": 8})
    plt.ylabel('Target Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix after Complete Iterations')
    plt.tight_layout()
    plt.show()
    print("Final Tesing Accuracy: " + str(round(acc*100,3)) +"%")
    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(round(time_dif,1)) + "s.")
    
data = numpy.load('/content/drive/My Drive/Projects/training_Final2.npy')
data_test = numpy.load('/content/drive/My Drive/Projects/validation_Final2.npy')
data_size = data.shape[0]
outputPermute = numpy.array(list(itertools.permutations([0,1,2,3])))

img_size = 128
img_size_flat = img_size*img_size
cut_size = int(img_size/2)
num_classes = 24
num_channels = 1
num_cut = 4

filter_size = 5
num_filters1 = 16
num_filters2 = 32
num_filters3 = 48
num_filters4 = 64
num_filters5 = 96
num_filters6 = 128
fc_size1 = 2048
fc_size2 = 512
fc_size3 = 128

testFeature, testTarget = getdata(outputPermute,data_test,data_test.shape[0])

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)

layer_conv1 = new_conv_layer(x_image,num_channels,filter_size,num_filters1,True)
layer_conv2 = new_conv_layer(layer_conv1,num_filters1,filter_size,num_filters2,True)
layer_conv3 = new_conv_layer(layer_conv2,num_filters2,filter_size,num_filters3,True)
layer_conv4 = new_conv_layer(layer_conv3,num_filters3,filter_size,num_filters4,True)
layer_conv5 = new_conv_layer(layer_conv4,num_filters4,filter_size,num_filters5,True)
layer_conv6 = new_conv_layer(layer_conv5,num_filters5,filter_size,num_filters6,True)
layer_flat, num_features = flatten_layer(layer_conv6)
layer_fc1 = new_fc_layer(layer_flat,num_features,fc_size1,True)
layer_fc2 = new_fc_layer(layer_fc1,fc_size1,fc_size2,True)
layer_fc3 = new_fc_layer(layer_fc2,fc_size2,fc_size3,True)
layer_output = new_fc_layer(layer_fc3,fc_size3,num_classes,False)

y_pred = tf.nn.softmax(layer_output)
y_pred_cls = tf.argmax(y_pred, axis=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_output, labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.Session()
session.run(tf.global_variables_initializer())

train_batch_size = 96
test_batch_size = 96
num_iterations = 100

itr_interval = 10

list_itr = (num_iterations*int((data_size)/(train_batch_size*itr_interval)))

accuracyList_train = numpy.zeros((list_itr,1))
accuracyList_test = numpy.zeros((list_itr,1))

optimize(list_itr)

numpy.save('/content/drive/My Drive/Projects/cnn_trainAccuracyList.npy',accuracyList_train)
numpy.save('/content/drive/My Drive/Projects/cnn_testAccuracyList.npy',accuracyList_test)

plot_graph(accuracyList_train,accuracyList_test,itr_interval)

print("Process Completed")