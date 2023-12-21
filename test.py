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


def loadDir2(action: str, data: list):  # read whole data in directory

    if action == 'golf':
        label = [b'00', b'AR', b'TB', b'BT', b'DS', b'IP', b'FT', b'FS']
    elif action == 'bowling':
        label = [b'00', b'AR', b'PS', b'DS', b'BT', b'FR', b'RL', b'FS']
    elif action == 'walking':
        label = [b'00', b'HS', b'LR', b'MD', b'TM', b'PS', b'TO', b'MS', b'TS']

    x_data, y_data = [], []
    data = loadData2(label=label, data=data)

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

    return x, y[2]


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

    # model = trainAction(action='walking', path='walking', epochs=200)
    # pred = predAction(action='golf', path='golf')

    data = [b'\x02\x00\n\x01USER01\xb0Db>\xdfq\x8f?hX3>\x8bC\xf2>P8\xd8=r\xabT?8\x0f\x8c>X\xec\x81>\x94"\xbe?\x80r\x83\xbc\x95\xf2\xa9\xbe\xd9\xf9\xe2>\xf5\xfe@\xbe\xe6\x9fO?\x80\x81\xc7<\x9c\xca\xbc?\xf4\x0f\x8a>[\xa0-\xbf\xb3\xc5\xb7=\xb1\x83:?T\'\n\xbd\xc0\xfdI>|E\xca?\xf8;o>\x8a\x8d\x17?!\xa6`>\x850-?&\x1a\xc2\xbe\x90\x93^>\x80\rl>\xb0\xd4\xd6=\\\xec3?\xda\xdcI\xbc\xea\xec5?6,\xf1<\xa8\x8f\x88\xbe\xe0h\x8b>\x00\xd5\xa9;\xd9.C?\x12\xcf\xb6\xbe\xc7\xe8\x03?\x0f/$>\x80\x14\xee<F\x9a\xc9? %`>`.;\xbf\xa5v\xbd\xbe\x1c\xf1\xb8>%\xcf\xe3>\xc0\xc3\x99=\xc5\'\xce?\xf0\xe3`>[;,\xbf\xe8\'\xde\xbeB\xf9\x17=\xfd\x1c\x19?\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x7f\x00\x00\x01\xd9LP@FS\x00\x03\x00', b'\x02\x00\n\x01USER01p\x7fc>\xca\x04\x8f?\xa8\xe93>\xf0\xe7\xf5>7\xd2\xd9=\x19\x88S?Ss\x8c>\x98\x88\x83>\xfe\xe4\xbd?@\x03\x89\xbc\x9e<\xab\xbeZ\xa3\xe6>\x12\x81;\xbe\xc2\xa9N?\x80\xcf\xe8<dr\xbc?\x0c\xc9\x8a>\\}1\xbf\xd1\x14\xb7=\xbb\xe36?^\x1b\xfa\xbc\x80\xa6M>\x9es\xc9?P\xc6r>\x87z\x13?i\x88[>\xc9N0?\xc3\xde\xc4\xbep\xb8c>\xc04k>\x80\x10\xd4=\x06~4?\x89Yj\xbc\xabO5?L.\x07=\x80t\x89\xbe0\xb8\x8a>\x00\xd8Y;"\xecA?3\xc4\xb5\xbe8>\x06?cp">@s\x10=[(\xc9?\xd8ha>\xa6\xdd;\xbf\xcfc\xc1\xbe;%\xb7>H\xab\xdf>\xa0:\xa2=\x04\xab\xcd?\xf0\xe2`>\xed\x08&\xbf\xe72\xe9\xbe3\xf5\xee\xbb}\x1b\x1c?\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x7f\x00\x00\x01\x10\x95T@FS\x00\x03\x00', b'\x02\x00\n\x01USER01p\x17g>I\xe5\x8e?\xe0\x834>(s\xf5>\x83\x82\xdd=.\x9bS?Np\x8c>@{\x86>a\xbf\xbd?\x00\xe5\x8b\xbc\xe7c\xac\xbe\xa9\xf6\xe7>HK?\xbes\xd5M?\x80\r\xf3<\xeaY\xbc?\xdc2\x8c>L;1\xbf\xcaO\xae=\xe0B7?b\x82\x01\xbdp\x7fT>\xc6U\xc9? \xe8q>\xe5\xbe\x14?\xacz^>\x8e\x15/?\r\x99\xc4\xbe \xbb_>\x10"k>\xa0 \xd6=\xec\x904?\xf9\x93\x88\xbc\x0f25?\x06\xc0\x10=P\x9f\x89\xbe\xb0\x88\x8a>\x00\x0ec;*!B?~4\xb6\xbeU\xef\x05?\xf2\x92 >@\x0c =4 \xc9?\xb0bc>\x0f\xaf;\xbf\x87M\xc1\xbe;\xbb\xb1>\xd2\xaa\xe4>@%\xb2=\xa8`\xcd? \xeaa>sw+\xbf_:\xe1\xbeJg\xb7<\x93\t\x19?\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x7f\x00\x00\x01A\xd9X@FS\x00\x03\x00', b'\x02\x00\n\x01USER01 ul>;\xd4\x8e?x\xcc4>JC\xf7>\x89\xd2\xd9="\xe2R?\xad\xf5\x8d>\x00\xe6\x8a>\x98\xa9\xbd?\x00\x01\x8d\xbcw\x98\xab\xbe\x13\xe7\xe8>q\xc1D\xbe\x98iM?@\xec\x11=\xfa\xb6\xbc?\x98\xe0\x8c>\xb0\x081\xbf\xe4{\xb1=\x1ci7?&\xb0\xfe\xbc\x90\xd0\\>\x0c[\xc9?hzq>\xc5o\x16?\x84g^>b0.?\xc9\xa4\xc2\xbe\xa0\xcfa>@\xfbj>\x90[\xd6=\xf5!5?|\x00\x8d\xbcB\x974?^\x89\x1b=\xc0T\x89\xbe\xd0\xb4\x8a>\x00\xec\xc2;\x90\xe4A?1\xc2\xb5\xbeQI\x06?4w">\xc0J?=\x94J\xc9?\xe8\xaeb>\xbe";\xbf\xebE\xc3\xbeg\xa3\xb1>\xc4\xdc\xe4>\x00K\xc1=\xd4y\xcd?\xa0Ic>\xcdz)\xbf\xbe\x1a\xe1\xbe\x16+\x88<\xc0S\x1b?\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x7f\x00\x00\x01\x84$]@FS\x00\x03\x00',
            b"\x02\x00\n\x01USER01\xf0\xbdq>\xe2\xd2\x8e?\xf0m4>d\xbe\xf8>\xe9@\xd9=\x87CR?>\x19\x8f>\x98\t\x8e>\xf2\x87\xbd?\x80\xa2\x8d\xbcb\xb3\xa9\xbe\x8e\x93\xe9>}\xc3G\xbe\xednM?\x80\x966=\x80\t\xbd?\xa0'\x8d>\t\x1e0\xbf4\xaa\xba=*(8?\xd8\xe5\xf8\xbc\x80\xa1g>\xb2s\xc9? \x9cp>\xb06\x17?\x90\x91U>y\xbf.?\x04\xb2\xc0\xbep\xc6`>\x10\x11h>0\x9e\xd6=\x0795?BR\xb7\xbc o4?7\xfb#=8\x8e\x88\xbe\xb8\xfe\x8a>\x00<|;\x1c\xd8@?51\xb5\xbe\x13\x0f\x08?\xe2]!>@\xf9h=4\x92\xc9?0\xa7a>k]<\xbfy\xc3\xc0\xbe~\x19\xb1>m^\xe3> 0\xd6=\xc0\xa3\xcd?P\xdab>kU*\xbf]\x0f\xe0\xbeyeb<\xa2\xc9\x1a?\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x7f\x00\x00\x01\xe0da@FS\x00\x03\x00", b'\x02\x00\n\x01USER01\x10\x11x>N\xca\x8e?H\xd63>\xed\x7f\xfa>\xac\xbb\xda=\x83kQ?d\xd5\x90>\x80\xbd\x90>%|\xbd?\x00\xdc\xa8\xbc\xb2\x97\xa9\xbe\x89\x08\xed>\x0c\xc8F\xbe\xf5\x85L?\x80g]=2Y\xbd?<\xb4\x8c>\xc7\xfd.\xbf\xef\xef\xc2=\xf3\x169?=-\xfc\xbc\xf0Dr>\xda\x82\xc9?(\xb5l>\x19n\x18?S\xf9S>\xd4S.?\xf2\xd0\xbe\xbe\xf0\x1fb>\xe0\xf4g>\x80\xaf\xd7=\xf9\x805?rr\xc9\xbcG\x1c4?*\x0f*=\xe0\xb9\x86\xbe@.\x8b>\x00\xe8\t;=\xf0>?\xe7M\xb5\xbe\x14\x98\n?8\x99">\x80n\x8a=\x1e\xe2\xc9? _`>d\xbb<\xbf\x86d\xba\xbe\x14\xc9\xb2>\x8a \xe6>\xa0\x9a\xea=\x9b\xce\xcd?P\x94`>\x8b\xbe,\xbf\xbf\xb1\xd8\xbebX\xa0<u\xb2\x1a?\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x7f\x00\x00\x01\x8b\xa5e@FS\x00\x03\x00', b"\x02\x00\n\x01USER01\xd0\xbf}>\x82\xd8\x8e?\x00\x902>\xac7\xf9>\xd5\x1e\xd8=x\x97Q?5J\x92>8\xa7\x93>\x94x\xbd?\x80\xcf\xd5\xbcI_\xac\xbek:\xf3>*\xc2F\xbet\x1cJ?`\xae\x85=\xe9\xc0\xbd? \xa0\x8b>\xd4/.\xbf\x1b\x81\xc6=\xf5\xc69?\xf5\x17\x02\xbd\x00\xba}>\xb0\xae\xc9?`\xd9h>\xcc\x89\x1a?\x95\xeeP>6\x85-?e\xca\xbb\xbeP\x08c>\xf0yh>\xc0i\xd7=?\x8d5?\x1f\xdc\xdc\xbc\x89\x0b4?\t\xa8(=p\xaf\x85\xbe\x987\x8b>\x00(t;\xd8\xba=?\x9d'\xb6\xbeu\xb2\x0b?\x83I&> #\xa0=;Z\xca?\x10\xb6^>\xd1\xde<\xbf\x94(\xb8\xbe\x15\xb3\xb3>G\xc2\xe6> \x08\x01>\xa1%\xce?\xe8*^>mM.\xbfRF\xd5\xbe\xb6\xc7\xd6<Z\x12\x1a?\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x7f\x00\x00\x01\x0c\xe9i@FS\x00\x03\x00"]
    loadDir2(action='golf', data=data)
