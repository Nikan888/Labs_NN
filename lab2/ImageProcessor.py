import xml.etree.ElementTree as Et
import glob
from PIL import Image
import numpy as np

tree = Et.parse('References.xml')
root = tree.getroot()

def getImages():

    trainHappy = root[0][0].text
    trainSad = root[0][1].text

    testHappy = root[1][0].text
    testSad = root[1][1].text

    imagesTrain = []
    for h in glob.iglob(trainHappy + "/*"):
        imagesTrain.append(np.asarray(convert(h)))

    #imagesTrain = np.array(imagesTrain)

    for s in glob.iglob(trainSad + "/*"):
        imagesTrain.append(np.asarray(convert(s)))

    imagesTrain = np.array(imagesTrain)

    imagesTest = []
    for h in glob.iglob(testHappy + "/*"):
        imagesTest.append(np.asarray(convert(h)))

    #imagesTest = np.array(imagesTest)

    for s in glob.iglob(testSad + "/*"):
        imagesTest.append(np.asarray(convert(s)))

    imagesTest = np.array(imagesTest)

    return imagesTrain, imagesTest

def convert(image):
    temp = Image.open(image)
    temp = temp.convert('1')
    A = np.array(temp)
    new_A = np.empty((A.shape[0], A.shape[1]), None)

    for i in range(len(A)):
        for j in range(len(A[i])):
            if A[i][j] == True:
                new_A[i][j] = 0
            else:
                new_A[i][j] = 1

    converted = new_A.flatten()
    return converted

def getAnswers():
    trainAnswers = root[0][2].text
    #text_file = open(trainAnswers + "\labels.txt", "r")
    #answers = text_file.read().split(',')
    answers = np.loadtxt(trainAnswers + "\labels.txt", comments="#", delimiter=",", unpack=False)
    return answers
