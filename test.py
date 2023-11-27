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

    if action == 'golf':
        label= [b'00', b'AR', b'TB', b'BT', b'DS', b'IP', b'FT', b'FS']
    elif action == 'bowling':
        label= [b'00', b'AR', b'PS', b'DS', b'BT', b'FR', b'RL', b'FS']
    elif action == 'walking':
        label= [b'00', b'HS', b'LR', b'MD', b'TM', b'PS', b'TO', b'MS', b'TS']

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

def loadDir2(action:str, data:list):  # read whole data in directory

    x_data, y_data = [], []
    data = loadData2(action,data=data )

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

    return x_data, y_data[2]

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


if __name__ == '__main__':

    # action = 'golf'
    action = 'bowling'
    # action = 'walking'

    # predModel(action, path=action+'_model.keras', x_test=x_test, y_test=y_test)

    # x_train, y_train, x_test, y_test = loadAction(action)

    data = [
            b'\x02\x00\n\x01USER21\xb8\xa3\xc7\xbe\xc8vy?\x00\xcdw\xbcS\xd1\x1e\xbf\xf5\xee\xaa\xbe^B)?X\x0e\x84>\xeat\x16\xbf\x84\x19\xb2?\x10\xed\x90=\x18\xdc\x12>\xa8\x7f5\xbf\xab\xc5\x13>\xd2\xdc,?t\xa6\xf4\xbe(BR?4x\xae\xbe\xa2\xbd\xa1>~\xd2l?Dh\x1c\xbd:@T\xbe\xac\x05|\xbf#3\x83?\xa8\xc2\x8a>ORC>Q\x96\xf3>\x0e\xa4U\xbf\xbd\xe6N>\xf4Z\xfd\xbe8\xa2\x84>@\x16H\xbd\xa5\xc5\xe8\xbecI1=ocb?\xfb\x85\xc6\xbd \xef\xb4\xbd\xa8\xad\x83>\x88\x0cg\xbe[\x99&\xbf \x02\x04\xbf\xb1r\r?\xc1\x07\x95=\xbc\x08\xf3\xbe\x10\x00<?\xbc\xe3\x99\xbeoH\xdf>\x0bI\x01\xbf@\xf5\x03\xbcN\xaa>?\xf3\xdf\x8c\xbfw\xc8\x88?\xb0aq>6\x9bq\xbd\x94\x07f\xbe\x07\x83d\xbe\x95[r?\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x7f\x00\x00\x01\x17T+@RL\x00\x03\x00',
            b'\x02\x00\n\x01USER21\xc0\xb9\xd2\xbe\xa0Lv?\xe0\xdf\x01\xbd\x95:\x1d\xbf\xd6\x81\xe8\xbe4\xcb"?\xb9\x0b\xe2=\xde\x7f\x1e\xbf!\xa4\xb1?`\xf5B=B3\x0e>\xaa;9\xbf\xd7\xbf!>pM(?\xf4\xe4\xf8\xbe6\xa7R?4\xae\xb1\xbe,\xec\xb7>\rWk?\xa0\xd8\xb2\xbc\r\x1e#\xbe\xd6\xff\x85\xbf\xfem\xa3?8Qo>\xf5I\xaf\xbe\xc6\x16\xc2\xbe\xb1\r\\?\xb6Mi<\xb0P\xfd\xbe\x00\xa0\x83>\xc0\x17A\xbdJ^\xd9\xbe\xb9\x01\xdc<\xd8&f?i\xaa\xd4\xbd\xb0\x83\x0c\xbeP\xe9p>4]\x89\xbe\x87\xbe\r\xbf\x1aR\x00\xbf|\xfa)?<\x1c\x14=\x04\xd8\xfd\xbe\n+=?\xec\xf7\x9d\xbe\x80\x9c\xd0>Jt\x10\xbf\xb0\xd0\xcf\xbc3\xb47?\xdeC\x8d\xbf\xee\x8c\xaf?\xf0\x08:>\x7f<\x03\xbe0\x08\x83\xbe\\\xef\x08\xbfa\x82K?\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x7f\x00\x00\x01C\x9c/@RL\x00\x03\x00',
            b'\x02\x00\n\x01USER21\x8c\xa2\xdb\xbe\xaa3x?\xc0\x7f\x0b\xbd\xa4\xd7*\xbf\x9f\xd7\xd0\xbe7X\x1c?\xae+\xfd=2Y%\xbf\x97\x98\xb1?@\xbb\xf1<N\xd6\x08>;f;\xbf\x80\xd7)>\xce\xa9%?\x98h\xf6\xbe\xc8\x17Q?\xecb\xb3\xbe6j\xc0>\x0c\x0bl?\xb3K\x89\xbcr\xd2\xba\xbd\xfa\xb8\x7f\xbfRr\xba?h\xe6K>P\x94\xe5\xbe\xd8\x9e\x8c\xbe\xb4$T?GhD>\x98j\xfe\xbe\x88\xad\x82> hs\xbd\xc06\xf0\xbe\xaa\x98/=^\x83_?\xbe{\x00\xbe@\x9a!\xbe@\xacn>\\\x0b\x8e\xbe9}\x04\xbf\xcc)\xfb\xbe\x8dD3?\xc4\x83\n=\x10\xbe\x00\xbf*\xbf<?\x0c\x7f\x9f\xbe1<\xbe>\xc7\xbd \xbf\xabX\x06\xbb\xb9\x12/?\xdas|\xbf\xbc=\xc8?(\xc9\x05>\xb5\\;>\xf4g\x99>\xa9\x1eC?\x9f<\x0b\xbf\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x7f\x00\x00\x01\x05\xcf3@FS\x00\x03\x00',
            b"\x02\x00\n\x01USER218\x05\xe5\xbe\x84\x00x?@\x0eN\xbd \xcb/\xbf\x00\x14\xdd\xbe\xef|\x14?e\xdb\x98=\xb2\x0c*\xbf\x0e\xb5\xb1?\x00Ut<R&\x08>th;\xbf\xd9\xc72>\x9e\x19%?\xec\xe7\xf2\xbe\x08;N?\xb4\xec\xb3\xbe]=\xb9>\xc1Hn?\xfc\xfd\xa6\xbc\xe2\x97D\xbd^\xd5k\xbf\x9a\x9f\xc3?\xf040>\xcf\xdc\t\xbf\x97R)\xbe=XC??7\xa2>,\xa7\xfe\xbe@\xf6\x81>\x807\x80\xbd\xe8a\xf4\xbe\xb0j =+\x1f^?\xf9\xa0\x08\xbe@\x12$\xbeP\xbfp>\xe4y\x88\xbe\xd3\xa9\x05\xbf\x94\xc9\xf1\xbe\xc7\x815?\xd0\xd8'=4\xff\xfd\xbe~\x0e:?\xb4\x99\xa0\xbed\x13\xc0>\x80Z%\xbf\x87\xbb\x0f=\xa8\xf8)?@QY\xbfj\xa4\xcf?`\x9c\xd0=jVc>\xc1\x02u>\xc9\xe4`?1\xa0\xb2\xbe\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x7f\x00\x00\x01w\x1a8@FS\x00\x03\x00",
            b"\x02\x00\n\x01USER218\x05\xe5\xbe\x84\x00x?@\x0eN\xbd \xcb/\xbf\x00\x14\xdd\xbe\xef|\x14?e\xdb\x98=\xb2\x0c*\xbf\x0e\xb5\xb1?\x00Ut<R&\x08>th;\xbf\xd9\xc72>\x9e\x19%?\xec\xe7\xf2\xbe\x08;N?\xb4\xec\xb3\xbe]=\xb9>\xc1Hn?\xfc\xfd\xa6\xbc\xe2\x97D\xbd^\xd5k\xbf\x9a\x9f\xc3?\xf040>\xcf\xdc\t\xbf\x97R)\xbe=XC??7\xa2>,\xa7\xfe\xbe@\xf6\x81>\x807\x80\xbd\xe8a\xf4\xbe\xb0j =+\x1f^?\xf9\xa0\x08\xbe@\x12$\xbeP\xbfp>\xe4y\x88\xbe\xd3\xa9\x05\xbf\x94\xc9\xf1\xbe\xc7\x815?\xd0\xd8'=4\xff\xfd\xbe~\x0e:?\xb4\x99\xa0\xbed\x13\xc0>\x80Z%\xbf\x87\xbb\x0f=\xa8\xf8)?@QY\xbfj\xa4\xcf?`\x9c\xd0=jVc>\xc1\x02u>\xc9\xe4`?1\xa0\xb2\xbe\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x7f\x00\x00\x01w\x1a8@FS\x00\x03\x00",
            b"\x02\x00\n\x01USER218\x05\xe5\xbe\x84\x00x?@\x0eN\xbd \xcb/\xbf\x00\x14\xdd\xbe\xef|\x14?e\xdb\x98=\xb2\x0c*\xbf\x0e\xb5\xb1?\x00Ut<R&\x08>th;\xbf\xd9\xc72>\x9e\x19%?\xec\xe7\xf2\xbe\x08;N?\xb4\xec\xb3\xbe]=\xb9>\xc1Hn?\xfc\xfd\xa6\xbc\xe2\x97D\xbd^\xd5k\xbf\x9a\x9f\xc3?\xf040>\xcf\xdc\t\xbf\x97R)\xbe=XC??7\xa2>,\xa7\xfe\xbe@\xf6\x81>\x807\x80\xbd\xe8a\xf4\xbe\xb0j =+\x1f^?\xf9\xa0\x08\xbe@\x12$\xbeP\xbfp>\xe4y\x88\xbe\xd3\xa9\x05\xbf\x94\xc9\xf1\xbe\xc7\x815?\xd0\xd8'=4\xff\xfd\xbe~\x0e:?\xb4\x99\xa0\xbed\x13\xc0>\x80Z%\xbf\x87\xbb\x0f=\xa8\xf8)?@QY\xbfj\xa4\xcf?`\x9c\xd0=jVc>\xc1\x02u>\xc9\xe4`?1\xa0\xb2\xbe\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x7f\x00\x00\x01w\x1a8@FS\x00\x03\x00",
            b'\x02\x00\n\x01USER21\x94\n\xe8\xbe\xfe\xa0w?\xe0HQ\xbd%\x9c.\xbf\xe9\xff\xdb\xbe\xea\x80\x16?\xff\xa7\x89=VH,\xbf\x9e\xa3\xb2?\x00\x84Y;l\x16\xf1=!\xc9;\xbf]\x9a.>!\xb5%?|\x08\xf0\xbe\xfcCL?\xc4\xb2\xb4\xbe\xb4\xeb\xa8> \x1bq?Y\x14\xe6\xbc\xe9\x99l\xbd\xb0Yh\xbf\x05\xfc\xc0?H\x10(>T\xa1\x0b\xbf?\x04\xfe\xbd\xcc\x05B?\x82\xe3\xab>\x14+\xff\xbe8d\x81>`6\x81\xbd\xf3@\xfb\xbe\x95\x9c:==\xbd[?\x1e\x04\x12\xbe\xf0)\x1f\xbe\xe0\xcao>$\x8d\x82\xbe_a\r\xbf\x83\xf4\xfa\xbe@\x1f,?D\xd9T=$\xaa\xf6\xbe\xceH7?\xf4\x8d\xa0\xbe^Q\xd2>2\xf6\x1c\xbf\xc9F|=\xf9\x05,?\x06&T\xbf\xcd\x02\xc9?P\x08\xad=\x89\x92\x92>\xd3\xc9y>\xc6Pc?\xac\x8b\x87\xbe\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x7f\x00\x00\x01<f<@FS\x00\x03\x00'
    ]

    x_data, y_data = loadDir2(action, data=data)

    print(0)

    # trainModel(action=action, epochs=500, x_train=x_train,
    #    y_train=y_train, x_test=x_test, y_test=y_test)
