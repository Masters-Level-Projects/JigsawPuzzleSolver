#####################################
########## Anurag Banerjee ##########
########### 2018CSM1007 #############
#####################################

import os
import numpy as np
import cv2
import time
import math


def normalize(X):
    X = X.astype(np.float32)
    mean = np.mean(X, axis=0, dtype=np.float32)
    sd = np.std(X, axis=0, dtype=np.float32)
    res = (X-mean)/sd
    res = res.astype(np.float16)
    return res


def combineXY(target, X, Y, count):
    data = np.zeros((X.shape[0], X.shape[1] + 1), dtype=np.float16)
    X = normalize(X)
    data[:, :-1] = X
    data[:, data.shape[1] - 1:data.shape[1]] = Y
    savepath = target + "data" + str(count)
    np.save(savepath, data)


def generate_img_mat(size, base, temp):
    response = int(input("\nAre you sure? This will delete previous data! Also, this is time consuming...\nif you want to continue, enter 1 (to quit, any other number): "))
    if response == 1:
        X = np.empty((0, 4096), dtype=np.uint8)
        Y = np.empty((0, 1), dtype=np.uint8)
        print("Generating data matrix X, Y")
        count = 0
        for path, subdirs, files in os.walk(base):
            for name in files:
                nextFile = os.path.join(path, name)
                count = count + 1
                # read Image and resize
                image = cv2.imread(nextFile)
                image = image.astype('uint8')
                    #cv2.imshow('Original Image', image)
                    #cv2.waitKey(0)
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    #cv2.imshow('Gray Image', gray_image)
                    #cv2.waitKey(0)
                resized_img = cv2.resize(gray_image, (size, size))
                # Now crop
                width_mid_pt = int(resized_img.shape[1]/2)
                height_mid_pt = int(resized_img.shape[0]/2)
                lefttop = resized_img[0:height_mid_pt, 0:width_mid_pt]
                righttop = resized_img[0:height_mid_pt, width_mid_pt:]
                leftbottom = resized_img[height_mid_pt:, 0:width_mid_pt]
                rightbottom = resized_img[height_mid_pt:, width_mid_pt:]

                img = lefttop.flatten()
                X = np.append(X, [img], axis=0)
                Y = np.append(Y, [[1]], axis=0)
                img = righttop.flatten()
                X = np.append(X, [img], axis=0)
                Y = np.append(Y, [[2]], axis=0)
                img = leftbottom.flatten()
                X = np.append(X, [img], axis=0)
                Y = np.append(Y, [[3]], axis=0)
                img = rightbottom.flatten()
                X = np.append(X, [img], axis=0)
                Y = np.append(Y, [[4]], axis=0)

                if count % 500 == 0:
                    print(str(count)+" images processed...")
                if count % 1000 == 0:
                    combineXY(temp, X, Y, count)
                    X = np.empty((0, 4096), dtype=np.uint8)
                    Y = np.empty((0, 1), dtype=np.uint8)

        combineXY(temp, X, Y, count)
    else:
        print("You chose wisely...!")


def prep_input(inloc, outloc):
    proc_data = None
    for path, subdirs, files in os.walk(inloc):
        for name in sorted(files):
            nextFile = os.path.join(path, name)
            #print("fullpath = "+nextFile)
            print("Reading file :"+nextFile)
            data = np.load(nextFile)
            data = data.astype(np.float16)
            if proc_data is None:
                proc_data = np.empty((0, data.shape[1]), dtype=np.float16)
            proc_data = np.append(proc_data, data, axis=0)
    print("\nSaving the processed data...")
    np.save(outloc+"data", proc_data)
    print("\nSaving the processed data...done")
    print("\nData ndarray of size "+str(int(round(proc_data.size/(1024 ** 2), 0)))+" MB created")
    print("\nShuffling the images...")
    blocksize = 4       # each image has 4 parts - when shuffling images, all blocksize parts must be together
    m, n = (proc_data.shape[0] // blocksize), proc_data.shape[1]
    reshaped_mat = proc_data.reshape(m, -1, n)
    np.random.shuffle(proc_data.reshape(m, -1, n))
    np.save(outloc + "data", proc_data)
    print("\nShuffling the images...done")


def gettimespent(sec):
    min = None
    hour = None
    if sec <= 60:
        return str(int(sec))+str(' seconds')
    elif sec > 60:
        min = sec / 60
        sec = min - math.floor(min)
        min = int(min - sec)
        sec = int(sec * 60)
    if min <= 60:
        return str(min)+str(' minutes ')+str(sec)+str(' seconds')
    elif min > 60:
        hour = min / 60
        min = hour - math.floor(hour)
        hour = int(hour - min)
        min = int(min * 60)
        return str(hour)+str(' hours ')+str(min)+str(' minutes ')+str(sec)+str(' seconds')


def main():
    while True:
        print("\n--------Pre-processing Menu----------")
        print("Enter 1 to generate images' matrix from images")
        print("Enter 2 to prepare RNN-LSTM input data matrix")
        ch = int(input("Enter 0 to quit: "))

        if ch == 1:
            start = time.time()
            generate_img_mat(size=128, base="./rawdata", temp="./tempdata/")
            end = time.time()
            timespent = gettimespent(end-start)
            print("\nPre-processing took: "+timespent)
        elif ch == 2:
            prep_input(inloc="./tempdata", outloc="./procdata/")
        elif ch == 0:
            break
        else:
            print("\nWrong choice! Try again.")


if __name__ == "__main__":
    main()
