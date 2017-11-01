import numpy as np
import _pickle as pickle
import matplotlib.pyplot as plt

#問題セットを取得する
def get_omniglot_questionset(use_data, use_label, QUESTION_NUM = 100, CHOICE_NUM = 5):
    """
    QUESTION_NUM : 総問題数
    CHOICE_NUM : 1問あたりの選択肢の数
    """
    question_set = []
    for i in range(QUESTION_NUM):
        N = np.arange(max(use_label))
        np.random.shuffle(N)
        #答えと選択肢のクラスをランダムに選ぶ
        ans_label,choice_label = N[:1],N[:CHOICE_NUM]

        ans_data = np.zeros((0,1,105,105))
        choice_data = np.zeros((0,1,105,105))
        #各クラスからデータをランダムに選ぶ
        #正解のクラス
        data = use_data[use_label==ans_label]
        np.random.shuffle(data)
        ans_data = np.append(ans_data,data[0].reshape((1,1,105,105)),axis=0)
        choice_data = np.append(choice_data,data[1].reshape((1,1,105,105)),axis=0)

        #不正解のクラス
        for label in choice_label[1:]:
            data = use_data[use_label==label]
            np.random.shuffle(data)
            choice_data = np.append(choice_data,data[0].reshape((1,1,105,105)),axis=0)

        #順番をランダムに入れ替える
        perm = np.random.permutation(CHOICE_NUM)
        choice_data = choice_data[perm]
        choice_label = choice_label[perm]

        question_set.append({"question_data":ans_data,"question_label":ans_label,"choice_data":choice_data,"choice_label":choice_label})
        
    return question_set

def show_questionset(questionset,line=5):
    #表示する行数
    M = line
    #表示する列数
    N = len(questionset[0]["question_data"])+len(questionset[0]["choice_data"])
    fig = plt.figure(figsize=(N*2,M*2))
    for i in range(M):
        ans_data,ans_label,choice_data,choice_label = \
                questionset[i]["question_data"],questionset[i]["question_label"],questionset[i]["choice_data"],questionset[i]["choice_label"]
        fig.add_subplot(M,N,1+i*N)
        plt.title("Question({})".format(ans_label[0]))
        #解答の表示
        plt.imshow(questionset[i]["question_data"][0][0],cmap="gray")
        plt.tick_params(labelbottom="off",bottom="off") # x軸の削除
        plt.tick_params(labelleft="off",left="off") # y軸の削除
        #選択肢の表示
        for j in range(N-1):
            fig.add_subplot(M,N,1+i*N+(j+1))
            plt.title("Choice{}({})".format(j+1,choice_label[j]))
            plt.imshow(questionset[i]["choice_data"][j][0],cmap="gray")
            plt.tick_params(labelbottom="off",bottom="off") # x軸の削除
            plt.tick_params(labelleft="off",left="off") # y軸の削除