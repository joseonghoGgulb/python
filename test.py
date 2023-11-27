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

    x_data, y_data = windowing(x_data, y_data)

    return x_data, y_data


def loadDir2(action: str, data: list):  # read whole data in directory

    if action == 'golf':
        action = [b'00', b'AR', b'TB', b'BT', b'DS', b'IP', b'FT', b'FS'], {
            0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 2, 7: 0}, 'golf'  # label, index, keyword
    elif action == 'bowling':
        action = [b'00', b'AR', b'PS', b'DS', b'BT', b'FR', b'RL', b'FS'], {
            0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 2, 7: 2}, 'bowling'
    elif action == 'walking':
        action = [b'00', b'HS', b'LR', b'MD', b'TM', b'PS', b'TO', b'MS', b'TS'], {
            0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2}, 'walking'

    x_data, y_data = [], []
    data = loadData2(action[0], data=data)

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

    return x_data, [action[1][y_data[0][2]]]


def loadAction(action: str):  # load data with input string
    if action == 'golf':
        action = [b'00', b'AR', b'TB', b'BT', b'DS', b'IP', b'FT', b'FS'], {
            0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 2, 7: 0}, 'golf'  # label, index, keyword
    elif action == 'bowling':
        action = [b'00', b'AR', b'PS', b'DS', b'BT', b'FR', b'RL', b'FS'], {
            0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 2, 7: 2}, 'bowling'
    elif action == 'walking':
        action = [b'00', b'HS', b'LR', b'MD', b'TM', b'PS', b'TO', b'MS', b'TS'], {
            0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2}, 'walking'

    x_train, y_train = loadDir(action[2], action[0])
    x_test, y_test = loadDir(action[2]+'_test', action[0])

    for i in range(0, len(y_train)):  # change label to index number
        y_train[i] = action[1][y_train[i]]

    for i in range(0, len(y_test)):  # change label to index number
        y_test[i] = action[1][y_test[i]]

    return x_train, y_train, x_test, y_test


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


def trainModel(action: str, epochs: int, x_train: list, y_train: list, x_test: list, y_test: list,):  # keras model function

    classCount = 3  # do not change
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
    plt.show()

    model.save(action+'_model.keras')


def predModel(action: str, path: str, x_test: list, y_test: list):  # keras prediction function

    model = keras.models.load_model(path)

    pred = model.predict(x_test)

    np.savetxt(action+'_pred.txt', pred)
    np.savetxt(action+'_test.txt', y_test)
    print(pred)


if __name__ == '__main__':

    # action = 'golf'
    action = 'bowling'
    # action = 'walking'

    x_train, y_train, x_test, y_test = loadAction(action)

    trainModel(action=action, epochs=500, x_train=x_train,
       y_train=y_train, x_test=x_train, y_test=y_train)

    # x_data, y_data = loadDir2(action, data=data)

    # print(x_data)
    # print(y_data)

    # predModel(action, path=action+'_model.keras', x_test=x_data, y_test=y_data)

