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

    # x_train, y_train, x_test, y_test = loadAction(action)

    data = [b'\x02\x00\n\x01USER21\x08P\xa3?\xf1\x16\x96?\xc0\xc8\x16>, )\xbfY5j\xbe\xdc\x87.?\x13\xa9\\>"s\xa3?\x9a4\xd6?(\x7f+>S\x15\xc6\xbb\x1b\xaf.\xbf\xa4\x01\xb9<j\x0b;?:\\\x89?\x84\x1f\x8d?\xd8`\x0f>=\xe0\x0c\xbf\xd6\x03\x1f\xbf/_\xd7>\x8b\xb3\xbb>\n\n\x94?>(\x91?H\x96\x93>\x04\x87\xaf\xbe\xef\xbc\xb4\xbe(\\R?\xa2.\x93\xbe*\xd6\xb8?\x00-\x83> \x8d:=\xe5g\'\xbf\x9c\xa2\x13\xbd\xb2\xa8@?]\x8d\x8c\xbd\n\xf0\xb4?\x08\x9a\x81>\xa8\xbcn>?\xa0J\xbf\xac\r&\xbc\x1aH\x1b?N\xb4\x97=.\x9e\x8e?\xfb\xdf\x8a?Phn>\x85\xec\x1e\xbd\xb1$\x1d\xbe|\xdc\x08\xbe7rz?h$\x89?D\x03\x96?\xc8\x05a>\xb5\xd7\xc3\xbd\xa8\xf5\x9d\xbeKh\x12\xber\x7fo?\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x7f\x00\x00\x01\xd0Xe=AR\x00\x03\x00', b"\x02\x00\n\x01USER21\n\x05\xa3?c\xfc\x95?\x80\xe4\x16>\xdf\x87(\xbfY\x9fo\xbe\xb4\xfa.?\xafgX>L\xee\xa2?^5\xd6?H\r,>\xa0)\x06\xbc_\xbc.\xbf\xa8\xf6\xab<\xc3\x00;?$3\x89?K\x89\x8c?0\xaf\x0c>\xa5\x82\x0c\xbf\x92\xd8 \xbf\xd4w\xd7>\xf8^\xb6>\x84\x95\x93?\xc9\xdb\x90?\xb0\xca\x93>\xbf\x8b\xaf\xbe\x15S\xb2\xbe\x02\x93R?r\xde\x94\xbef\xe6\xb8?P\xf5\x82>\xe0\x878=e\xed'\xbf\x9c\xda\x10\xbd\x007@?\x97`\x8c\xbd\x9c\xc5\xb4?\x80\xb5\x81>\xd8&n>\x037K\xbf\xe3\xc1?\xbc\xee\x8f\x1a?U\xc8\x93=(-\x8e?,\x91\x8a?0\xe5l>\xac\\#\xbdsi*\xbe\x83\x11\x07\xbe\xfa\xf3y?\x80\x0b\x89?\x9e\xa6\x95?\x90h_>\xf5f\xca\xbdY\xa7\xa2\xbe#\xfb\x1a\xbesIn?\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x7f\x00\x00\x01=z\xf8=AR\x00\x03\x00", b'\x02\x00\n\x01USER21d\xf9\xa2?\xf1\xa6\x95?`\x14\x17>"\xd9&\xbf/*y\xbe3B0?\xef\xc6Q>\xdc3\xa2?\x96\x04\xd6?\xf8\xcc->P\x9f\xfc\xbb\xb3\xc1.\xbf\xa3\x1a\xb2<\xb1\xfa:?V\xed\x88?n\x1d\x8c? A\x0c>d\xfc\x0b\xbf\xcf\xfc \xbfT+\xd7>\xe0\xd4\xb7>&\x17\x93?\x92n\x90?@\\\x94>\xb6x\xae\xbe0\xc0\xb1\xbe\x8c\xc2R?\xae\xc3\x95\xbe\xbe\x07\xb9?`\x9a\x82>\xc0\xdb6=Ju(\xbf\xc0\x9b\r\xbd\xfe\xc6??n\xca\x8a\xbdx\xa9\xb4? 6\x81>\xd0\xeel>\xa0\xc7K\xbfG6\x8c\xbc\xbc\xe6\x19? \xb6\x8b=\xe2\xcf\x8d?\xaf\x17\x8a?p\x96l>\xb0\x01&\xbd;\xd5-\xbea\x1d\x0c\xbe\x0b\xa0y?\xf6\xc4\x88?\xd3+\x95?H\xb4_>\xd8,\xc0\xbd7B\xa2\xbe$\x87\x1b\xbe\xd6vn?\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x7f\x00\x00\x01`w@>AR\x00\x03\x00', b"\x02\x00\n\x01USER21:\xaa\xa2?\xe4\x82\x95?\xf8\xe9\x19>8B&\xbf(bt\xbe\x1b\xbc0?c`X>\xecP\xa1?~\xe6\xd5?\xd0<1>i\xca\x1a\xbc\xb1\xd1.\xbf\xf8\xf4\xb7<\xfc\xe8:?:|\x88?\xa5&\x8b?\xf8\xb6\x0c>\x84\xcf\n\xbf\x91\x1d#\xbf\x96\xbe\xd5>y\x8a\xb5>@\x8b\x92?\xcc\xa2\x8f?\x10\x18\x95>\x0f\xdb\xaa\xbe\x98c\xb0\xbe|VS?\xc3C\x98\xbe\xca\r\xb9?\xf0\xa0\x82>@\xfa6=\xf5\xfa'\xbf\x81\x9a\r\xbd\xa15@?\xea\x97\x89\xbd`\x8b\xb4?\xc8\xd5\x80>@pl>\x95FL\xbfR\xac\xb3\xbc\xad;\x19?\x95\x86\x89=\xe2q\x8d?\x80:\x89?\x182n>C\xf2\x17\xbd\n\xf7,\xbep\x1c\xfd\xbdb&z?.`\x88?\x80A\x94?X\x98`>\x1c\x8c\xbe\xbd\xa4?\xa2\xbe\xf7\x95#\xbe/&n?\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x7f\x00\x00\x01\xe2w\x82>AR\x00\x03\x00",
            b"\x02\x00\n\x01USER21\xd6\xf7\xa1?s\xb8\x95? \xfe\x1d>\xf1\x10'\xbf^Um\xbeI\xd4/?@\xdea>*\x90\xa0?d\xce\xd5?\xb8)7>\xc5n\x17\xbc\xaf*/\xbf\x1b\xc4\xc7<\xb4\x91:?\xba\xe0\x87?\x18\x80\x89?\xb8Y\x15>\xff\x02\x04\xbf\xd4\xb5)\xbf\xd0H\xdb>\x18\xb7\xaa>\x14\xd1\x90?\xdb\xbf\x8e?\x80*\x98>2\x08\xa5\xbex\xe8\xa7\xbe\x01\x9bU?F\x95\x9b\xbe^\xdf\xb8?\xa8\xcc\x82>\xa0H<=\x0f<'\xbf%o\x17\xbd6\xce@?\x84\xb9\x8b\xbd\xd4\x81\xb4?\xe8\xf4\x80>\xe8xl>\xba\xf4L\xbf\xcb\xd4\xc4\xbc\xf3>\x18?Cf\x8d=\\\xd3\x8c?j\x12\x88?\x98\x89w>z\xba\x10\xbd\xdf\xb9;\xbe;F\xba\xbd'kz?\\\xfa\x86?\x119\x93?P\xd4e>\xd0r\xc2\xbdY\xb1\xa2\xbe3\x1f3\xbe\xf3Rm?\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x7f\x00\x00\x01\xd5\x88\xa4>AR\x00\x03\x00", b'\x02\x00\n\x01USER21\x94\xdf\xa1?\xda\x00\x96?\x98#\'>\\4&\xbf\xd3nd\xbe\xfb$0?\x11\xcbp>\xa8\xf2\x9f?mu\xd5?\xb0*?>{\xd1\xa8\xbb\xfa\x84.\xbf\xc2!\xf1<3#;?\x18r\x84?\xef\xe2\x8a?(\xb4\x1e>\xcf1\x08\xbf\xee\x16&\xbf\xea\x9a\xd3>\xac/\xb5>l\xa6\x8c?_\x9f\x8f?\xe4t\x9a>\xb7\xae\xa8\xbe=\x7f\xa9\xbe\x06pU?\x0c\xcb\x96\xbe\x18\x9e\xb8?\x18\x8a\x82>\xc0\x8b@=\x9e\x90&\xbf\x18\xd6*\xbd\x96:A?;\xac\x93\xbd"^\xb4?\xc8\x8e\x80>\x98\xb5l>.RM\xbf\n\xac\xeb\xbc\xa7\xaa\x17?u\x9e\x8f=x\xd5\x89?j\xdf\x88?\x18~\x7f>*\x9c\xe5\xbc\xb4\x8d\x16\xbek\xdb\xc6\xbd\x8f\xe4{?*\x04\x83?=\x87\x94?\xa0Ti>\xe8\xael\xbd\xf1\xc0\xa3\xbe~4\x16\xbe\n,o?\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x7f\x00\x00\x01S\xba\xc6>AR\x00\x03\x00', b'\x02\x00\n\x01USER21\x18\xf3\xa0?\xd2\xd3\x95?@\x7f/>\xb6(&\xbf+\x95a\xbe\xee\xa9/?\x06ry>\x104\x9f?\xd8\x15\xd5?\xc0\xd8H>\xe9X\x16\xbb)\xb0-\xbftM!=-\xcb;?\xf8\xda~?8\xef\x8f?p\xb6%>\x84@\x15\xbfR,\x1e\xbf\x1d\x00\xcb>-3\xb2>\x1a\xb9\x87?\xb4\x1e\x94?\xb0A\xa0>5j\xb2\xbe4!\xa1\xbeb\xcaY?H\xd0q\xbe\x98l\xb8?\x98\xa0\x82>`\xe8D=\xc5@&\xbf0\xe7>\xbd\xfaaA?\xfe\x17\x97\xbd\xce\\\xb4?x:\x80>\x88Lm>\xa0gM\xbf\x06\x0f\n\xbd\x94q\x17?\xeey\x92=\xe6\x89\x83?\xf3\x1c\x8e?\x10.\x85>>:!\xbdI\xeb&\xbe@\x1b\x16\xbe\xbe\x91y?\xb4\xbd~?Eg\x9a?\xd0.t>\x16\x1d\x8e\xbd\x1f\xf3\xa1\xbe:M`\xbe\xf5\x9fk?\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x7f\x00\x00\x01I\xb9\xe8>AR\x00\x03\x00']

    x_data, y_data = loadDir2(action, data=data)

    print(x_data)
    print(y_data)

    predModel(action, path=action+'_model.keras', x_test=x_data, y_test=y_data)

    # trainModel(action=action, epochs=500, x_train=x_train,
    #    y_train=y_train, x_test=x_test, y_test=y_test)
