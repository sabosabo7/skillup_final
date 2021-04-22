import numpy as np


def loaddata():
    """
    データを読む関数
    """
    test_data = np.load("../1_data/integ_data7.npy")
    return test_data


def accuracy(func_predict, test_data):
    """
    精度を計算する関数
    label_pred : numpy 1D array
    """

    # データ数のチェック
    data_size = len(test_data)
    if data_size < 9000:
        error = "label_predのサイズが足りていません"
        print(data_size, error)
        print("Test loss:", error)
        print("Test accuracy:", error)
        return
    elif data_size > 9000:
        error = "label_predのサイズが多すぎます"
        print(data_size, error)
        print("Test loss:", error)
        print("Test accuracy:", error)
        return

    test_label = np.load("../1_data/integ_label7.npy")
    batch_size = 500

    minibatch_num = np.ceil(data_size / batch_size).astype(int)  # ミニバッチの個数

    li_loss = []
    li_accuracy = []
    li_num = []
    index = np.arange(data_size)

    for mn in range(minibatch_num):
        print(mn)
        mask = index[batch_size * mn : batch_size * (mn + 1)]
        data = test_data[mask]
        label = test_label[mask]
        loss, accuracy = func_predict(data, label)
        print(loss, accuracy)

        li_loss.append(loss)
        li_accuracy.append(accuracy)
        li_num.append(len(data))

    test_loss = np.dot(li_loss, li_num) / np.sum(li_num)
    test_accuracy = np.dot(li_accuracy, li_num) / np.sum(li_num)

    print("Test loss:", test_loss)
    print("Test accuracy:", test_accuracy)
    return