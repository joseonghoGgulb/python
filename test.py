import struct
import numpy as np
import math
from tensorflow import keras
import os
import matplotlib.pyplot as plt
import pandas as pd


def unpackBytes(data: bytes, l: int = 4, t: str = 'f'):  # convert byte data to float data
    length = int(len(data)/l)

    tmp = []
    for i in range(0, length):
        tmp.append(struct.unpack(t, data[0+l*i:l+l*i])[0])

    return tmp


def loadData(name: str, label: list):  # read data file
    with open(name, "rb") as f:
        origin = f.read()

    data = origin.split(b'\x02\x00')[1:]
    output = []
    for i in data:
        tmp = byteToList(i)
        try:
            tmp['label'] = label.index(tmp['label'])
            output.append(tmp)
        except:
            print('[error] label: '+str(tmp['label']))

    return output


def loadData2(label: list, data: list):  # read data file

    output = []
    for i in data:
        i = i.split(b'\x02\x00')[1]
        tmp = byteToList(i)
        try:
            tmp['label'] = label.index(tmp['label'])
            output.append(tmp)
        except:
            print('[error] label: '+str(tmp['label']))

    return output


def byteToList(data: bytes):  # chang byte data to float data
    dataSize = struct.unpack('h', data[0:2])
    userId = data[2:8]
    pelvis = unpackBytes(data[8:36])
    head = unpackBytes(data[36:64])
    handR = unpackBytes(data[64:92])
    handL = unpackBytes(data[92:120])
    footR = unpackBytes(data[120:148])
    footL = unpackBytes(data[148:176])
    controllerR = unpackBytes(data[176:204])
    controllerL = unpackBytes(data[204:232])
    threadmill = unpackBytes(data[232:248])
    eye = unpackBytes(data[248:264])
    lip = unpackBytes(data[264:268])
    time = unpackBytes(data[268:272])
    label = data[272:274]
    crc = data[274:275]
    etx = data[275:277]

    return {'pelvis': pelvis, 'head': head, 'handR': handR, 'handL': handL, 'footR': footR, 'footL': footL, 'controllerR': controllerR, 'controllerL': controllerL,  'time': time, 'label': label}


# convert quaternion data set. it contain 7 float. output data contain 2 float.
def sevenToTwo(data: list):
    output = []
    output.append(data[0]**2+data[1]**2+data[2]**2)

    x = data[3]/math.sin(math.acos(data[6]))
    y = data[4]/math.sin(math.acos(data[6]))
    z = data[5]/math.sin(math.acos(data[6]))

    output.append(x**2+y**2+z**2)

    return output


# process of data converting. quaternion data set => featual data set.
def demensionReduction(data: list):
    output = []
    for i in data:
        try:
            i['pelvis'] = sevenToTwo(i['pelvis'])
            i['head'] = sevenToTwo(i['head'])
            i['handR'] = sevenToTwo(i['handR'])
            i['handL'] = sevenToTwo(i['handL'])
            i['footR'] = sevenToTwo(i['footR'])
            i['footL'] = sevenToTwo(i['footL'])
            i['controllerR'] = sevenToTwo(i['controllerR'])
            i['controllerL'] = sevenToTwo(i['controllerL'])
            output.append(i)
        except:
            print('[error] evaluation : '+str(i)+'\n')

    return output


def derivative(data: list):  # derivated by time interval
    out = []
    for i in range(1, len(data)):
        tmp = {}
        for j in data[i]:
            if j != 'time' and j != 'label':
                deri = np.array(data[i][j])-np.array(data[i-1][j])
                deri = deri/data[i]['time'][0]

                tmp[j+'_der'] = list(deri)

        tmp['time'] = data[i]['time']
        tmp['label'] = data[i]['label']

        out.append(tmp)

    return out


# concatenate lists. it contain derivatived data
def concatData(data1: list, data2: list, data3: list):

    data = []
    for i in range(0, len(data3)):
        tmp = data3[i]
        tmp.update(data2[i+1])
        tmp.update(data1[i+2])
        data.append(tmp)

    return data


def loadDir(path: str, label: list):  # read whole data in directory

    x_data, y_data = [], []
    file_list = os.listdir(path)
    for file_name in file_list:
        data = loadData(os.path.join(path, file_name), label=label)

        data = demensionReduction(data)

        data_1 = derivative(data)
        data_2 = derivative(data_1)
        data = concatData(data, data_1, data_2)

        # split orignal data by using label tag .  input data is x . ouput data is y.
        x, y = [], []
        for i in data:
            tmpX = []
            for j in i:
                if j == 'label':
                    tmpY = i[j]
                else:
                    for k in i[j]:
                        tmpX.append(k)

            x.append(tmpX)
            y.append(tmpY)

        x_data.append(x)
        y_data.append(y)

    y_data = labelingMeta(input=y_data, label=label)

    x_data, y_data = windowing(x_data, y_data)

    return x_data, y_data


def labelingMeta(input: list, label: list):
    y_data = input
    for i in range(0, len(input)):
        index = set(input[i])
        for j in range(0, len(input[i])):
            if j == 0:
                index.remove(input[i][j])

            elif input[i][j] in index:
                index.remove(input[i][j])
                y_data[i][j] += len(label)

    return y_data


def trainAction(action: str, path: str, epochs: int):  # load data with input string
    if action == 'golf':
        label = [b'00', b'AR', b'TB', b'BT', b'DS', b'IP', b'FT', b'FS']
        classCount = 16
    elif action == 'bowling':
        label = [b'00', b'AR', b'PS', b'DS', b'BT', b'FR', b'RL', b'FS']
        classCount = 16
    elif action == 'walking':
        label = [b'00', b'HS', b'LR', b'MD', b'TM', b'PS', b'TO', b'MS', b'TS']
        classCount = 18
    x_data, y_data = loadDir(path, label=label)

    xlen = int(len(x_data)*0.5)
    ylen = int(len(y_data)*0.5)

    x_train = x_data[0:xlen]
    y_train = y_data[0:ylen]
    x_test = x_data[xlen:-1]
    y_test = y_data[ylen:-1]

    model = trainModel(action=action, epochs=epochs, x_train=x_train,
                       y_train=y_train, x_test=x_test, y_test=y_test, classCount=classCount)

    return model


def predAction(action: str, path: str):  # load data with input string
    if action == 'golf':
        label = [b'00', b'AR', b'TB', b'BT', b'DS', b'IP', b'FT', b'FS']
    elif action == 'bowling':
        label = [b'00', b'AR', b'PS', b'DS', b'BT', b'FR', b'RL', b'FS']
    elif action == 'walking':
        label = [b'00', b'HS', b'LR', b'MD', b'TM', b'PS', b'TO', b'MS', b'TS']

    x_data, _ = loadDir(path, label=label)

    pred = predModel(
        action=action,  x_test=x_data, path=action+'_model.keras')

    return pred


def windowing(x: list, y: list, windowSize: int = 5, windowIndex: int = 3):  # split dataset by window

    tmpx = []
    for i in x:
        tmp = []
        for j in i:
            if len(tmp) < windowSize:
                tmp.append(j)
            else:
                tmpx.append(tmp)
                tmp = []

    tmpy = []
    for i in y:
        tmp = []
        for j in i:
            if len(tmp) < windowSize:
                tmp.append(j)
            else:
                tmpy.append(tmp[int(windowIndex)])
                tmp = []

    return tmpx, tmpy


def trainModel(action: str, epochs: int, x_train: list, y_train: list, x_test: list, y_test: list, classCount: int):  # keras model function

    featureCount = 49  # do not change
    windowSize = 5  # do not change

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(48, activation='elu',
              input_shape=(windowSize, featureCount)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(.2))
    model.add(
        keras.layers.Bidirectional(keras.layers.GRU(
            48, return_sequences=True))
    )
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(.2))
    model.add(keras.layers.Bidirectional(
        keras.layers.GRU(32)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(.2))
    model.add(keras.layers.Dense(classCount, activation='softmax'))

    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
                  optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    hist = model.fit(x_train, y_train, epochs=epochs,
                     validation_data=(x_test, y_test))

    df = pd.DataFrame(hist.history)  # display result of training
    df.plot()
    plt.savefig('save.jpg')

    model.save(action+'_model.keras')

    return model


def predModel(action: str, path: str, x_test: list):  # keras prediction function

    model = keras.models.load_model(path)

    pred = model.predict(x_test)

    np.savetxt(action+'_pred.txt', pred)


if __name__ == '__main__':

    model = trainAction(action='walking', path='walking', epochs=200)
    # pred = predAction(action='golf', path='golf')
