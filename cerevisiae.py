
import random
import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Dense, Dropout, Conv1D, Input,MaxPooling1D,Flatten,LeakyReLU,Activation,concatenate,Reshape,GRU,LSTM,Add,Multiply#, attention
from keras.layers import BatchNormalization
# from keras.optimizers import SGD
from tensorflow.keras.optimizers import SGD
# from keras import regularizers
from keras.metrics import binary_accuracy
from sklearn.metrics import confusion_matrix,recall_score,matthews_corrcoef,roc_curve,roc_auc_score,auc
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import losses
import tensorflow as tf
import keras
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "1";
np.random.seed(seed=21)
tf.__version__
keras.__version__


def analyze(temp, OutputDir):
    trainning_result, validation_result, testing_result = temp;
    file = open(OutputDir + '/result3.txt', 'w')
    index = 0
    for x in [trainning_result, validation_result, testing_result]:
        title = ''
        if index == 0:
            title = 'training_'
        if index == 1:
            title = 'validation_'
        if index == 2:
            title = 'testing_'
        index += 1;
        file.write(title +  'results\n')
        for j in ['sn', 'sp', 'acc', 'MCC', 'AUC', 'precision', 'F1', 'lossValue']:
            total = []
            for val in x:
                total.append(val[j])
            file.write(j + ' : mean : ' + str(np.mean(total)) + ' std : ' + str(np.std(total))  + '\n')

        file.write('\n\n______________________________\n')
    file.close();
    index = 0
    for x in [trainning_result, validation_result, testing_result]:
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        i = 0
        for val in x:
            tpr = val['tpr']
            fpr = val['fpr']
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            # plt.plot(fpr, tpr, lw=1, alpha=0.3,label='ROC fold %d (AUC = %0.2f)' % (i+1, roc_auc))
            i += 1
        print;
        # plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Random', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')

        # plt.xlim([-0.05, 1.05])
        # plt.ylim([-0.05, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver operating characteristic curve')
        # plt.legend(loc="lower right")

        title = ''

        if index == 0:
            title = 'training_'
        if index == 1:
            title = 'validation_'
        if index == 2:
            title = 'testing_'

        # plt.savefig( OutputDir + '/' + title +'ROC.png')
        # plt.close('all');

        index += 1;


def scheduler(epochs, lr):
  if epochs < 10:
    return lr
  else:
    return lr * tf.math.exp(-0.1)

    
def check_sequence(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    
    return out

def calculate(sequence):

    X = []
    dictNum = {'A' : 0, 'U' : 0, 'C' : 0, 'G' : 0};

    for i in range(len(sequence)):

        if sequence[i] in dictNum.keys():
            dictNum[sequence[i]] += 1;
            X.append(dictNum[sequence[i]] / float(i + 1));

    return np.array(X)

def dataProcessing_onehot(path,T):

    data = pd.read_csv(path);
    alphabet = np.array(['A', 'G', 'U', 'C','N'])
    X = [];
    data = data.values
    for line in range(0, len(data), 2):
        seq_data = []
        seq = data[line][0]
        for i in range(len(seq)):
            if seq[i] == 'A':
                seq_data.append([1,0,0,0])
            if seq[i] == 'U':
                seq_data.append([0,1,0,0])
            if seq[i] == 'C':
                seq_data.append([0,0,1,0])
            if seq[i] == 'G':
                seq_data.append([0,0,0,1])
            if seq[i] == 'N':
                seq_data.append([0,0,0,0])
                
        X.append(np.array(seq_data));
    if T == "P":
        y = [1]*len(X)
    if T == "N":
        y = [0]*len(X)
    X = np.array(X);
    y = np.array(y, dtype = np.int32);
    return X, y;

def dataProcessing_NCP(path, T):
    dataset = pd.read_csv(path,header=None)
    X = []
    for i in dataset.iloc[:, 0]:
        if ">" in i:
            continue
        seq_data = []
        for j in range(len(i)):
            if i[j] == "A":
                seq_data.append([1, 1, 1])
            elif i[j] == "C":
                seq_data.append([0, 1, 0])
            elif i[j] == "G":
                seq_data.append([1, 0, 0])
            elif i[j] == "T":
                seq_data.append([0, 0, 1])
            elif i[j] == "U":
                seq_data.append([0, 0, 1])
            else:
                seq_data.append([0, 0, 0])
        X.append(seq_data)
    if T == "P":
        y = [1] * len(X)
    if T == "N":
        y = [0] * len(X)
    X = np.array(X);
    y = np.array(y, dtype=np.int32);
    return X, y;


def prepareData(PositiveCSV, NegativeCSV):

    Positive_X_hot, Positive_y_hot = dataProcessing_onehot(PositiveCSV, "P");
    Negitive_X_hot, Negitive_y_hot = dataProcessing_onehot(NegativeCSV, "N");
    Positive_X_ncp, Positive_y_ncp = dataProcessing_NCP(PositiveCSV, "P");
    Negitive_X_ncp, Negitive_y_ncp = dataProcessing_NCP(NegativeCSV, "N");

    Positive_X = np.concatenate([Positive_X_hot, Positive_X_ncp], axis=-1)
    Negitive_X = np.concatenate([Negitive_X_hot, Negitive_X_ncp], axis=-1)
    Positive_y = Positive_y_hot
    Negitive_y = Negitive_y_ncp
    return Positive_X, Positive_y, Negitive_X, Negitive_y
def split_data(X):
    hot = X[:,:,:4]
    ncp = X[:,:,4:]

    return hot, ncp

def shuffleData(X, y):
    index = [i for i in range(len(X))]
    random.shuffle(index)
    X = X[index]
    y = y[index]
    return X, y;


def Att(inputs):
    V = inputs
    QK = Dense(64)(inputs)
    QK = Activation(activation='relu')(QK)
    MV = Multiply()([V, QK])
    return MV

def cnn_gru_attention_lstm():
    input_shape = (31,7)
    inputs = Input(shape = input_shape)

    conv0 = Conv1D(filters=32, kernel_size=5,strides=1)(inputs)
    normLayer0 = BatchNormalization()(conv0);
    act0 = Activation(activation='relu')(normLayer0)

    gru=GRU(64,return_sequences=True)(act0)

    conv2 = Conv1D(filters=64, kernel_size=5,strides=1)(act0)
    normLayer2 = BatchNormalization()(conv2);
    pool2 = MaxPooling1D(pool_size = 2)(normLayer2)
    dropoutLayer1 = Dropout(0.35)(pool2)
    act2 = Activation(activation='relu')(dropoutLayer1)
    res=concatenate([act2,gru],axis=1)

    x = Flatten()(res)

    # gru = GRU(64, return_sequences=True)(inputs)
    # lstm = LSTM(64, return_sequences=True)(inputs)
    # gru_lstm = concatenate([gru, lstm],axis=1)
    # print(gru_lstm.shape)
    # print(inputs.shape)
    # print(act2.shape)
    # cnn_gru_lstm = concatenate([act2, gru_lstm],axis=1)
    # x = Flatten()(cnn_gru_lstm)

    # lstm = LSTM(64, return_sequences=True)(inputs)
    # cnn_lstm = concatenate([act2, lstm], axis=1)
    # print(cnn_lstm.shape)
    # outputs=Dense(64, activation='relu')(inputs)
    # outputs=concatenate([act2,outputs],axis=1)
    # print(outputs.shape)
    # res = Add()([cnn_lstm, outputs])
    # x = Flatten()(res)

    conv3 = Conv1D(filters=64, kernel_size=5,strides=1)(act0)
    normLayer4 = BatchNormalization()(conv3);
    pool4 = MaxPooling1D(pool_size = 2)(normLayer4)
    dropoutLayer2 = Dropout(0.35)(pool4)
    act4 = Activation(activation='relu')(dropoutLayer2)
    res_2 = concatenate([act4, gru], axis=1)
    # gru=GRU(64,return_sequences=True)(res_2)
    # res_gru=Att(gru)
    res_2=Att(res_2)
    a = Flatten()(res_2)
    comb = concatenate([x, a], axis=1)
    
    a1 = keras.layers.Lambda(lambda comb: comb[:,0:384], output_shape=(384,))(comb)
    a2 = keras.layers.Lambda(lambda comb: comb[:,384:], output_shape=(384,))(comb)

    a1 = Dense(8, activation='relu')(a1)
    a2 = concatenate([a2, a1])
    a2 = Dense(8, activation='relu')(a2)
    a3 = concatenate([a1, a2])
    a3 = Dense(8, activation='relu')(a3)
    a4 = concatenate([a2, a3])
    a4 = Dense(8, activation='relu')(a4)
    a5 = concatenate([a1, a4])
    a5 = Dense(8, activation='relu')(a3)
    a6 = concatenate([a2, a5])
    a6 = Dense(8, activation='relu')(a6)
    a = concatenate([a1, a2], axis=1)
    a = concatenate([a, a3], axis=1)
    a = concatenate([a, a4], axis=1)
    a = concatenate([a, a5], axis=1)
    a = concatenate([a, a6], axis=1)

    output = Dense(1, activation= 'sigmoid')(a)

    model = Model(inputs = inputs, outputs = output)
    opt=SGD(learning_rate=0.001, momentum = 0.95)
    model.compile(loss='binary_crossentropy', optimizer= opt, metrics=[binary_accuracy]);
    
    return model


def calculateScore(X, y, model, folds):
    
    score = model.evaluate(X,y)
    pred_y = model.predict(X)

    accuracy = score[1];

    tempLabel = np.zeros(shape = y.shape, dtype=np.int32)

    for i in range(len(y)):
        if pred_y[i] < 0.5:
            tempLabel[i] = 0;
        else:
            tempLabel[i] = 1;
    confusion = confusion_matrix(y, tempLabel)
    TN, FP, FN, TP = confusion.ravel()

    sensitivity = recall_score(y, tempLabel)
    specificity = TN / float(TN+FP)
    MCC = matthews_corrcoef(y, tempLabel)

    F1Score = (2 * TP) / float(2 * TP + FP + FN)
    precision = TP / float(TP + FP)

    pred_y = pred_y.reshape((-1, ))

    ROCArea = roc_auc_score(y, pred_y)
    fpr, tpr, thresholds = roc_curve(y, pred_y)
    lossValue = None;

    y_true = tf.convert_to_tensor(y, np.float32)
    y_pred = tf.convert_to_tensor(pred_y, np.float32)
    
    # plt.show()
    
    lossValue = losses.binary_crossentropy(y_true, y_pred)#

    return {'sn' : sensitivity, 'sp' : specificity, 'acc' : accuracy, 'MCC' : MCC, 'AUC' : ROCArea, 'precision' : precision, 'F1' : F1Score, 'fpr' : fpr, 'tpr' : tpr, 'thresholds' : thresholds, 'lossValue' : lossValue}

def funciton(PositiveCSV, NegativeCSV, OutputDir, folds):

    Positive_X, Positive_y, Negitive_X, Negitive_y = prepareData(PositiveCSV, NegativeCSV)
    
    random.shuffle(Positive_X);
    random.shuffle(Negitive_X);

    Positive_X_Slices = check_sequence(Positive_X, folds);
    Positive_y_Slices = check_sequence(Positive_y, folds);

    Negative_X_Slices = check_sequence(Negitive_X, folds);
    Negative_y_Slices = check_sequence(Negitive_y, folds);

    trainning_result = []
    validation_result = []
    testing_result = []
    
    for test_index in range(folds):

        test_X = np.concatenate((Positive_X_Slices[test_index],Negative_X_Slices[test_index]))
        test_y = np.concatenate((Positive_y_Slices[test_index],Negative_y_Slices[test_index]))

        validation_index = (test_index+1) % folds;

        valid_X = np.concatenate((Positive_X_Slices[validation_index],Negative_X_Slices[validation_index]))
        valid_y = np.concatenate((Positive_y_Slices[validation_index],Negative_y_Slices[validation_index]))
        # print("wangping",test_X.shape)
        # print(valid_X.shape)

        start = 0;

        for val in range(0, folds):
            if val != test_index and val != validation_index:
                start = val;
                break;

        train_X = np.concatenate((Positive_X_Slices[start],Negative_X_Slices[start]))
        train_y = np.concatenate((Positive_y_Slices[start],Negative_y_Slices[start]))
        print(train_X.shape)
        for i in range(0, folds):
            if i != test_index and i != validation_index and i != start:
                tempX = np.concatenate((Positive_X_Slices[i],Negative_X_Slices[i]))
                tempy = np.concatenate((Positive_y_Slices[i],Negative_y_Slices[i]))
                train_X = np.concatenate((train_X, tempX))
                train_y = np.concatenate((train_y, tempy))
        print(np.shape(tempX),np.shape(train_X))
        test_X, test_y = shuffleData(test_X,test_y);
        valid_X,valid_y = shuffleData(valid_X,valid_y)
        train_X,train_y = shuffleData(train_X,train_y);
        
        print(np.shape(train_X), np.shape(valid_X), np.shape(test_X))
        # np.save('D:/RNA 假尿苷实验/修改模型/ZayyuNet-main/ZayyuNet-main/Output_musculus_Pseudo/chunk_folds/'+str(test_index)+'_'+'x_test',test_X)
        # np.save('D:/RNA 假尿苷实验/修改模型/ZayyuNet-main/ZayyuNet-main/Output_musculus_Pseudo/chunk_folds/'+str(test_index)+'_'+'y_test',test_y)
        # np.save('D:/RNA 假尿苷实验/修改模型/ZayyuNet-main/ZayyuNet-main/Output_musculus_Pseudo/chunk_folds/'+str(test_index)+'_'+'valid_X',valid_X)
        # np.save('D:/RNA 假尿苷实验/修改模型/ZayyuNet-main/ZayyuNet-main/Output_musculus_Pseudo/chunk_folds/'+str(test_index)+'_'+'valid_y',valid_y)
        # np.save('D:/RNA 假尿苷实验/修改模型/ZayyuNet-main/ZayyuNet-main/Output_musculus_Pseudo/chunk_folds/'+str(test_index)+'_'+'x_train',train_X)
        # np.save('D:/RNA 假尿苷实验/修改模型/ZayyuNet-main/ZayyuNet-main/Output_musculus_Pseudo/chunk_folds/'+str(test_index)+'_'+'y_train',train_y)

        model = cnn_gru_attention_lstm();

        result_folder = OutputDir
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        model_results_folder=result_folder
        
        early_stopping = EarlyStopping(monitor='val_binary_accuracy', patience= 30, restore_best_weights=True)
        model_check = ModelCheckpoint(filepath = OutputDir + "/model" + str(test_index+1) +".h5", monitor = 'val_binary_accuracy', save_best_only=True, save_weights_only=True)
        
        reduct_L_rate = tf.keras.callbacks.LearningRateScheduler(scheduler)
        
        cbacks = [model_check, early_stopping,reduct_L_rate]

        
        history = model.fit(train_X, train_y, batch_size = 32, epochs = 50, validation_data = (valid_X, valid_y),callbacks = cbacks);

        # plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        # plt.title('model loss')
        # plt.ylabel('loss')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'test'], loc='upper left')

        trainning_result.append(calculateScore(train_X, train_y, model, folds));
        validation_result.append(calculateScore(valid_X, valid_y, model, folds));
        testing_result.append(calculateScore(test_X, test_y, model, folds));

    temp_dict = (trainning_result, validation_result, testing_result)
    analyze(temp_dict, OutputDir);


# NegativeCSV = r".\cerevisiae_negitive_314.txt"
# PositiveCSV = r".\cerevisiae_positive_314.txt"
#
# OutputDir = r"D:\RNA 假尿苷实验\王苹假尿苷实验\cerevisiae_result"
# funciton(PositiveCSV, NegativeCSV, OutputDir, 10);


def data_(file_name):
    if "cerevisiae" in file_name:
        lens = 31
        na = "cerevisiae"
        f_1 = open(".\cerevisiae_positive_314.txt", "r")
        f_2 = open(".\cerevisiae_negitive_314.txt", "r")
    elif "musculus" in file_name:
        lens = 21
        na = "musculus"
        f_1 = open(".\musculus_positive_472.txt", "r")
        f_2 = open(".\musculus_negitive_472.txt", "r")
    else:
        lens = 21
        na = "sapiens"
        f_1 = open(".\sapiens_positive_495.txt", "r")
        f_2 = open(".\sapiens_negitive_495.txt", "r")
    l, n = [], []
    for (text, text_2)  in zip(f_1.readlines(), f_2.readlines()):
        w = text.strip(">")
        l.append(">H."+na+"_" + w.strip("\n"))
        n.append(">H."+na+"_" + text_2.strip("\n").strip(">"))
    l = l + n
    file_names = l[0::2]
    site = ["Yes", "No"]
    # data = []
    # for i in file_names:
    #     if "P" == i:
    #         site = site[0]
    #     else:
    #         site = site[1]
    #     data.append([i, lens, site])

    data = [[i , lens, site[0]] if "P" in i  else [i , lens, site[1]] for i in file_names ]
    return data

# 导入tk
from tkinter import *
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk




def make_tk():

    # 创建tk窗口
    window = tk.Tk()
    window.title('my window')
    window.geometry('1200x730+0+0')
    var = tk.StringVar()
    l = tk.Label(window, bg="white", width=30, height=70, text='Please select the data set you want：', font=("宋体", 19), wraplength=300)
    l.pack(side="left")


    def data_show():
        import tkinter.font as tf
        text.delete("1.0", "end")
        font1 = tf.Font(family='微软雅黑', size=12)
        if "cerevisiae" in var.get():
            f_1 = open(".\cerevisiae_positive_314.txt", "r")
            f_2 = open(".\cerevisiae_negitive_314.txt", "r")
        elif "musculus" in var.get():
            f_1 = open(".\musculus_positive_472.txt", "r")
            f_2 = open(".\musculus_negitive_472.txt", "r")
        else:
            f_1 = open(".\sapiens_positive_495.txt", "r")
            f_2 = open(".\sapiens_negitive_495.txt", "r")
        for i in f_1.readlines():
            text.insert(END, i)
        for i in f_2.readlines():
            text.insert(END, i)
        text.config(font=font1)
    # 按钮1及其功能
    r1 = tk.Radiobutton(window,command=data_show, text='cerevisiae', variable=var, value='cerevisiae',  width=27,height=1,fg='red',font=('宋体',20))
    r1.pack(pady=10)

    r2 = tk.Radiobutton(window,command=data_show, text='musculus', variable=var, value='musculus' , width=27,height=1,fg='red',font=('宋体',20))
    r2.pack(pady=10)

    r3 = tk.Radiobutton(window,command=data_show, text='sapiens', variable=var, value='sapiens',  width=27,height=1,fg='red',font=('宋体',20))
    r3.pack(pady=10)
    r1.select()# 默认选择

    # 触发功能即按下按钮后想要程序做什么
    def print_selection():
        l.config(text='Is predicting ' + var.get() + " dataset sites for you.")
        if var.get() == "cerevisiae":
            NegativeCSV = r".\cerevisiae_negitive_314.txt"
            PositiveCSV = r".\cerevisiae_positive_314.txt"
            OutputDir = r"D:\RNA 假尿苷实验\王苹假尿苷实验\cerevisiae_result"
        elif var.get() == "musculus":
            NegativeCSV = r".\musculus_negitive_472.txt"
            PositiveCSV = r".\musculus_positive_472.txt"
            OutputDir = r"D:\RNA 假尿苷实验\王苹假尿苷实验\musculus_result"
            # messagebox.showinfo("第二步")
        else:
            NegativeCSV = r".\sapiens_negitive_495.txt"
            PositiveCSV = r".\sapiens_positive_495.txt"
            OutputDir = r"D:\RNA 假尿苷实验\王苹假尿苷实验\sapiens_result"
        # funciton(PositiveCSV, NegativeCSV, OutputDir, 10);

        '''
        表格
        '''
        # 创建tk窗口
        win1 = tk.Tk()
        win1.title('数据显示：')
        win1.geometry('600x500+200+200')
        data = data_(var.get())

        tree = ttk.Treeview(win1, columns=('name', 'len', 'site'), show="headings",
                            displaycolumns="#all", height=19)
        tree.column("name", anchor="center")
        tree.column("len", anchor="center")
        tree.column("site", anchor="center")
        tree.heading('name', text="Sequence",)
        tree.heading('len', text="Number of nucleotides")
        tree.heading('site', text="Site")
        style = ttk.Style()
        style.configure("Treeview.Heading", font=("宋体", 20))
        for itm in data:
            tree.insert("", tk.END, values=itm)
        tree.pack(expand=1)

    a = tk.PanedWindow(sashrelief=tk.SUNKEN, background="#1DF5DF", width=200)
    a.pack()
    btn1 = tk.Button(a,text='Confirm',command=print_selection, width=40,height=1,fg='black',font=('宋体',20))
    a.add(btn1)
    # 显示数据内容
    scr1 = tk.Scrollbar(window)
    scr1.pack(side='right', fill=tk.Y)  # 垂直滚动条
    text = tk.Text(window, width=80, height=10)
    text.pack(side='left', fill=tk.BOTH, expand=True)
    text.config(yscrollcommand=scr1.set)  # 滚动设置互相绑定
    scr1.config(command=text.yview)  # 滚动设置互相绑定

    window.mainloop()

make_tk()






