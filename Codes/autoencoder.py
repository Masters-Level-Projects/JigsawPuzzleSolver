#Name- Anurag Singh Mehta
#Entry Number- 2016CSB1111

from keras.datasets import mnist
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K 
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import numpy as np
import itertools
print(K.image_data_format())

import numpy as np
import matplotlib.pyplot as plt
import itertools
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

def plot_history(history):  # function to plot loss and accuracy using logs of training 
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]
    
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return 
    
    ## As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)
    
    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    ## Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:    
        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def getdata(op,data,data_size):
    itemsTaken = data
    img_size = size
    cut_size = cut
    num_classes = 24
    num_cut = 4
    #image_size_flat = 
    imageTaken = np.reshape(itemsTaken,(data_size,img_size,img_size))
    x_LT = imageTaken[:,0:cut_size,0:cut_size]
    x_RT = imageTaken[:,0:cut_size,cut_size:img_size]
    x_LD = imageTaken[:,cut_size:img_size,0:cut_size]
    x_RD = imageTaken[:,cut_size:img_size,cut_size:img_size]
    y_true = np.zeros((data_size,num_classes))
    x_items = np.zeros((data_size,img_size,img_size))
    items_img = np.zeros((num_cut,cut_size,cut_size))
    #tempIndex = numpy.array([0,1,2,3])
    list_permute = np.array(list(itertools.permutations([0,1,2,3])))
    for indexImg in range(data_size):
        if indexImg % num_classes == 0: np.random.shuffle(list_permute)
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
    #x_items = np.reshape(x_items,(data_size,img_size_flat))
    return(x_items,y_true)
  
#temp2 = np.load("/content/gdrive/My Drive/validation.npy")
#op = np.array(list(itertools.permutations([0,1,2,3])))
#X,Y = getdata(op,temp2,len(temp2))

size = 128 # dimension of the images
cut = 64
#(x_train, _), (x_test, _) = mnist.load_data()
'''
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(y_train.shape)
'''
input_img = Input(shape=(size, size, 1))  # adapt this if using `channels_first` image data format
#print(input_img.shape)

x = Conv2D(128, (3, 3), activation='relu', padding='same')(input_img)

x = MaxPooling2D((2, 2), padding='same')(x)
#print("encoded 1",x.shape)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)

x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

print("encoded shape",encoded.shape)
# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
#print("decoded 1",x.shape)
x = UpSampling2D((2, 2))(x)
#print("decoded 1",x.shape)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
#print("decoded 1",x.shape)
x = UpSampling2D((2, 2))(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)

#print("decoded 2",x.shape)
x = Conv2D(128, (3, 3), activation='relu',padding='same')(x)
#print("decoded 1",x.shape)
x = UpSampling2D((2, 2))(x)
#print("decoded 3",x.shape)
decoded = Conv2D(1, (5, 5), activation='sigmoid', padding='same')(x)


print("decoded shape",decoded.shape)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy',metrics=['accuracy'])

x_train = np.load("/content/gdrive/My Drive/training_Final2.npy")  # path to the training dataset 
x_test = np.load("/content/gdrive/My Drive/validation_Final2.npy")	# path to the test dataset
op = op = np.array(list(itertools.permutations([0,1,2,3])))

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train1, y_true1 = getdata(op,x_train,len(x_train))
x_train1 = np.reshape(x_train1, (len(x_train1), size,size, 1))
x_train2 = np.reshape(x_train, (len(x_train), size,size, 1))

x_test1 , y_true2 = getdata(op,x_test,len(x_test))
x_test1 = np.reshape(x_test1, (len(x_test1), size, size, 1))
x_test2 = np.reshape(x_test, (len(x_test), size,size, 1))

#x_train = np.reshape(x_train, (len(x_train), 32, 32, 1))  # adapt this if using `channels_last` image data format
#x_test = np.reshape(x_test, (len(x_test), 32, 32, 1))  # adapt this if using `channels_last` image data format


history = autoencoder.fit(x_train1, x_train2,epochs=50,batch_size=32,shuffle=True,validation_data=(x_test1, x_test2),verbose =1)

def show_imgs(x_test, decoded_imgs=None, n=10):
    plt.figure(figsize=(20, 8))
    for i in range(n):
        ax = plt.subplot(2, n, i+1)
        plt.imshow(x_test[i].reshape(size,size))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if decoded_imgs is not None:
            ax = plt.subplot(2, n, i+1+n)
            plt.imshow(decoded_imgs[i].reshape(size,size))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()

decoded_imgs = autoencoder.predict(x_test1)
plot_history(history)	
print ("input (upper row)\ndecoded (bottom row)")
show_imgs(x_test2, decoded_imgs)

def part_err(part1,part2):
  #print("part1 shape: ",part1.shape)
  #print("part2 shape: ",part2.shape)
  error = part1-part2
  #print(error)
  error = np.square(error)
  t = error.sum()
  return float(t)/256


def cal_ind(part,in_img,part_ind,error_mat):
  #print("in_img shape: ",in_img.shape)
  lt1 = in_img[0:cut,0:cut]
  rt1 = in_img[0:cut,cut:size]
  ld1 = in_img[cut:size,0:cut]
  rd1 = in_img[cut:size,cut:size]
  index = -1
  err_arr = np.zeros(4)
  err_arr[0] = part_err(part,lt1)
  err_arr[1] = part_err(part,rt1)
  err_arr[2] = part_err(part,ld1)
  err_arr[3] = part_err(part,rd1)
  min = 100000
  #error_mat = np.zeros((4,4))
  for j in range(4):
    error_mat[part_ind][j] = err_arr[j]
    '''
    if(err_arr[j] < min):
      min = err_arr[j]
      index = j
    '''
  return error_mat
    
def find_min(mat,rem_index,av_frames):
  #print(mat.shape[0])
  r = -1
  c= -1
  min = 100000
  for i in rem_index:
    for j in av_frames:
      if(mat[i][j] < min):
        min = mat[i][j]
        r=i
        c=j
  return [r,c]
    
  
def cal_err(xin,xout):		# function to calculate accuracy 
  
  xin = np.reshape(xin,(len(xin),size,size))
  xout = np.reshape(xout,(len(xin),size,size))
  #print("xin shape: ",xin.shape)
  #print("xout shape: ",xout.shape)
  error = 0
  for ind in range(len(xin)):
    in_img = xin[ind]
    out_img = xout[ind]
    lt2 = out_img[0:cut,0:cut]
    rt2 = out_img[0:cut,cut:size]    
    ld2 = out_img[cut:size,0:cut]
    rd2 = out_img[cut:size,cut:size]
    #print("lt2 shape: ",lt2.shape)
    index_sf = np.zeros(4)
    av_frames = [0,1,2,3]
    rem_index = [0,1,2,3]
    error_mat = np.zeros((4,4))
    error_mat = cal_ind(lt2,in_img,0,error_mat)
    error_mat = cal_ind(rt2,in_img,1,error_mat)
    error_mat = cal_ind(ld2,in_img,2,error_mat)
    error_mat = cal_ind(rd2,in_img,3,error_mat)
    
    '''
    index[0] = cal_ind(lt2,in_img,0)
    index[1] = cal_ind(rt2,in_img,1)
    index[2] = cal_ind(ld2,in_img,2)
    index[3] = cal_ind(rd2,in_img,3)
    '''
    flag = 0
    #print(index)
    for _ in range(4):
      [r,c] = find_min(error_mat,rem_index,av_frames)
      index_sf[r] = c
      #np.delete(error_mat,r,0)
      #np.delete(error_mat,c,1)
      rem_index.remove(r)
      av_frames.remove(c)
    #print(index_sf)
    
    for i in range(4):
      if(index_sf[i] != i):
          flag = flag+1
    if(flag >2):
       error = error + 1
  return float(error)/len(xin)
       
total_err = cal_err(x_test2,decoded_imgs)
print("Total accuracy: ",(1-total_err)*100,"%")
  

