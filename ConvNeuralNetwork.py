import glob
import numpy as np
from PIL import Image
import os
import random
import tensorflow as tf
import time

IMAGE_SIZE = 224
IMAGE_FOLDER = "C:/Users/toon1/Documents/Private Reading Spring 2018/AllImages" #

RANDOM_SEED = 1334
TRAINING_PERCENT = 0.7
VALID_PERCENT = 0.0
LEARNING_RATE = 0.0001
MAX_STEPS = 600
BATCH_SIZE = 64

def readImages():
    images = []
    labels = []

    os.chdir(IMAGE_FOLDER)

    for filename in glob.glob("chestnut_*.jpg"):
        im = Image.open(filename)
        npIm = np.array(im)

        images.append(npIm)
        labels.append([1])

    for filename in glob.glob("not_*.jpg"): 
        im = Image.open(filename)
        npIm = np.array(im)

        images.append(npIm)
        labels.append([0])

    return images, labels


def splitData(images, labels):
    x = [[] for i in range(3)]
    y = [[] for i in range(3)]

    indices = list(range(len(images)))
    random.shuffle(indices)

    trainSize = int(len(images) * TRAINING_PERCENT)
    validSize = int(len(images) * VALID_PERCENT)

    for i in range(len(images)):
        index = indices[i]
        if i < trainSize:
            x[0].append(images[index])
            y[0].append(labels[index])
        elif i < trainSize + validSize:
            x[1].append(images[index])
            y[1].append(labels[index])
        else:
            x[2].append(images[index])
            y[2].append(labels[index])

    return x, y


def batch(x, y, batchSize, nextIndex):
    batchX = []
    batchY = []

    n = len(x)
    for i in range(batchSize):
        batchX.append(x[nextIndex])
        batchY.append(y[nextIndex])
        nextIndex = (nextIndex + 1) % n

    return batchX, batchY, nextIndex



def trainNetwork(splitImages, splitLabels):
    width = splitImages[0][0].shape[0]
    x = tf.placeholder(dtype=tf.float32, shape=[None, width, width, 3])
    y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

    conv1 = tf.layers.conv2d(inputs=x, filters=96, kernel_size=[7,7], strides=2, padding="same", activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=2)

    conv2 = tf.layers.conv2d(inputs=pool1, filters=256, kernel_size=[5, 5], strides=2, padding="same", activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=2)

    conv3 = tf.layers.conv2d(inputs=pool2, filters=384, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)

    conv4 = tf.layers.conv2d(inputs=conv3, filters=384, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)

    conv5 = tf.layers.conv2d(inputs=conv4, filters=256, kernel_size=[15, 15], padding="same", activation=tf.nn.relu)
    pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[3, 3], strides=2)
    # print(pool5.get_shape())  # TODO use this to find the correct shape for the next line -- in mine it is [None, 7, 7, 256]
    pool5_flat = tf.reshape(pool5, shape=[-1,6*6*256]) # TODO update if this is the wrong shape

    dense1 = tf.layers.dense(inputs=pool5_flat, units=4096, activation=tf.nn.relu)
    dense2 = tf.layers.dense(inputs=dense1, units=4096, activation=tf.nn.relu)

    output = tf.layers.dense(inputs=dense2, units=1)
    predict = tf.nn.sigmoid(output)

    cost = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output)
    trainer = tf.train.AdamOptimizer(LEARNING_RATE)
    train_step = trainer.minimize(cost)

    # create the session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    step = 0
    elapsedTime = 0
    nextIndex = 0
    print("Step,Train,Test,Loss")
    while step < MAX_STEPS:
        step += 1

        # get the batch
        batchX, batchY, nextIndex = batch(splitImages[0], splitLabels[0], BATCH_SIZE, nextIndex)

        startTime = time.perf_counter()
        sess.run(train_step, feed_dict={x: batchX, y: batchY})
        endTime = time.perf_counter()
        elapsedTime += endTime - startTime

        if step % 50 == 0:
            trainMatrix, trainAcc = calcResults(x, predict, sess, splitImages[0], splitLabels[0])

            # validMatrix, validAcc = calcResults(x, predict, sess, splitImages[1], splitLabels[1])

            testMatrix, testAcc = calcResults(x, predict, sess, splitImages[2], splitLabels[2])

            print(step,trainAcc, testAcc, cost,sep=",")


def calcResults(x, predict, sess, images, labels):
    matrix = [[0, 0], [0, 0]]

    start = 0
    end = min(len(labels), start + BATCH_SIZE)
    correct = 0
    while start < end:
        imagesSub = images[start:end]
        labelsSub = labels[start:end]

        p = sess.run(predict, feed_dict={x: imagesSub})
        matrixSub, _ = calcPerformance(p, labelsSub)

        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                matrix[i][j] += matrixSub[i][j]

                if i == j:
                    correct += matrixSub[i][j]

        start = end
        end = min(len(labels), start + BATCH_SIZE)


    return matrix, correct / len(labels)


def calcPerformance(rawPredictions, labels):
    predictions = []
    for rawPred in rawPredictions:
        if rawPred[0] >= 0.5:
            predictions.append(1)
        else:
            predictions.append(0)

    total = 0
    correct = 0
    matrix = [[0, 0], [0, 0]]
    for i in range(len(labels)):
        pred = predictions[i]
        actual = int(labels[i][0])

        matrix[actual][pred] += 1

        if actual == pred:
            correct += 1

        total += 1

    return matrix, correct / total


def main():
    random.seed(RANDOM_SEED)

    # process the data
    images, labels = readImages()
    splitImages, splitLabels = splitData(images, labels)

    # create and train the network
    trainNetwork(splitImages, splitLabels)

main()
