from glob import glob
import numpy as np
import cv2
import re
import _pickle as pickle
import sys

#使用するフォルダ名と出力するファイル名を指定する
#i.e.　python get_data.py DATA/images_background_small1 DATA/small1
args = sys.argv

#globで全てのファイル名を取得
files = glob("{}/**/**/**".format(args[1]))

#1つの文字に1つのidを対応付る
filenames = []
for file in files:
    gr = re.search(r"^[^/]+/([^/]+/[^/]+)",file)
    if not gr.group(1) in filenames:
        filenames.append(gr.group(1))

chara2id = {fname:np.array([i]) for i,fname in enumerate(filenames)}

#==========データの取得と変換===========#
#環境によるが,smallで数分,largeで数十分かかる
train_data = np.zeros((0,1,105,105))
train_label = np.zeros((0,1))
test_data = np.zeros((0,1,105,105))
test_label = np.zeros((0,1))

for file in files:
    img = cv2.imread(file)
    #グレースケール変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #データ整形
    img = (255-gray)/255
    img = img.reshape((1,1,105,105))
    
    #データの追加
    #各クラス５つずつをテストデータに、
    #それ以外はトレーニングデータにする
    gr = re.search(r"_([0-9]{2}).png",file)
    gr2 = re.search(r"^[^/]+/([^/]+/[^/]+)",file)
    if int(gr.group(1)) <= 5:
        test_data = np.append(test_data,img,axis=0) 
        test_label = np.append(test_label,chara2id[gr2.group(1)]) 
    else:
        train_data = np.append(train_data,img,axis=0)
        train_label = np.append(train_label,chara2id[gr2.group(1)]) 
        
train_data = np.array(train_data,dtype=np.float32)
train_label = np.array(train_label,dtype=np.int32)
test_data = np.array(test_data,dtype=np.float32)
test_label = np.array(test_label,dtype=np.int32)

np.savez_compressed("{}.npz".format(args[2]),train_data=train_data,train_label=train_label,test_data=test_data,test_label=test_label)