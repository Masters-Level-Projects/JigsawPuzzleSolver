#Name- Subhranil Bagchi
#Entry Number- 2018CSY0002

import time
import cv2
import glob
import numpy


def process1():
    outputPath = "dataset/"
    inputPath = "inputdir"
    count = 0
    print("Preprocessing Images")
    st = time.time()
    for path, subdirs, files in os.walk(inputPath):
        for name in files:
            nextFile = os.path.join(path, name)
            image = cv2.imread(nextFile,cv2.IMREAD_GRAYSCALE)
            midY = int(image.shape[0]/2)
            midX = int(image.shape[1]/2)
            if midX>midY:
                imgLength = (midY*2)-1
                X_not = midX-midY
                Y_not = midY-midY
            else:
                imgLength = (midX*2)-1
                X_not = midX-midX
                Y_not = midY-midY
            image = image[Y_not:Y_not+imgLength,X_not:X_not+imgLength]
            count = count + 1
            image = cv2.resize(image,(128,128))
            cv2.imwrite(outputPath+format(count,"0>8d")+".jpg",image)
            if count%1000==0: print(str(count)+" Image Completed")
    print("Preprocessing Completed. Took " + str(time.time()-st))


def process2():
    inputPath = "dataset/*.jpg"
    itemIndex = 0
    no_features = (128 * 128)
    samples = len(glob.glob(inputPath))
    dataMatrix = numpy.zeros((samples, no_features), dtype='uint8')
    print("Creating Image Matrix")
    category = 0
    for imgName in glob.glob(inputPath):
        image = cv2.imread(imgName)
        image = image[:, :, 0]
        image = image.reshape(image.shape[0] * image.shape[1])
        # imgNorm = numpy.asarray(image,dtype='double')
        dataMatrix[itemIndex] = image
        itemIndex = itemIndex + 1
        if itemIndex % 1000 == 0: print(str(itemIndex) + " Image Completed")
    print("Image Matrix Completed")
    """
    print("Standardizing")
    for i in range((dataMatrix.shape[1])):
        m = numpy.mean(dataMatrix[:,i])
        s = numpy.std(dataMatrix[:,i])
        dataMatrix[:,i] = (dataMatrix[:,i]-m)/s
        if i%10==0: print(str(i)+" Features Completed")
    print("Standardizing Completed")
    """
    numpy.save('features.npy', dataMatrix)
    print("File Saved")


def process3():
    print("Importing Data Matrix")
    dataMatrix = numpy.load('features.npy')
    print("Data Matrix Imported")
    numpy.random.shuffle(dataMatrix)
    trainSize = int(dataMatrix.shape[0] * 0.75)
    print("Spliting")
    training, validation = dataMatrix[:trainSize, :], dataMatrix[trainSize:, :]
    print("Saving")
    numpy.save('training_Final2.npy', training)
    numpy.save('validation_Final2.npy', validation)
    print("Saved")


if __name__ == "__main__":
    process1()
	process2()
    process3()