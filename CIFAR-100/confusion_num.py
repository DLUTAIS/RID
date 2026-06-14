from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import json

classes = ['A1', 'A2', 'H1', 'L1', 'L2', 'M1', 'O1', 'S1', 'S2', 'W1', 'X1']  # 数据类别y
models = ['shufflenetV2_110']  # 我的数据子路径,用于下面读取数据
annot_file = './labels/labels.txt'  # 实际的数据类别y_true
json_path = './labels/pred.txt'  # 预测的数据类别y_pred

# 读取数据真实类别y_true
read_file = open(annot_file, 'r', encoding="utf - 8")
y_true = []
for line in read_file:
    line_num = line.split('\t')
    if len(line_num) > 1:
        for index in range(len(line_num)-1):
            y_true.append(line_num[index])
read_file.close()

# 读取数据预测类别y_pred
for model in models:
    read_file = open(json_path, 'r', encoding="utf - 8")
    y_pred = []
    for line in read_file:
        line_num = line.split('\t')
        if len(line_num) > 1:
            for index in range(len(line_num) - 1):
                y_pred.append(line_num[index])
        print(len(y_pred))
    read_file.close()

    # 计算混淆矩阵，FP，FN，TP，TN，PRECISION，RECALL
    cm = confusion_matrix(y_true, y_pred, labels=None, sample_weight=None)

    FP = sum(cm.sum(axis=0)) - sum(np.diag(cm))  # 假正样本数
    FN = sum(cm.sum(axis=1)) - sum(np.diag(cm))  # 假负样本数
    TP = sum(np.diag(cm))  # 真正样本数
    TN = sum(cm.sum().flatten()) - (FP + FN + TP)  # 真负样本数
    SUM = TP + FP
    PRECISION = TP / (TP + FP)  # 查准率，又名准确率
    RECALL = TP / (TP + FN)  # 查全率，又名召回率


    def plot_confusion_matrix(cm, savename, title='Confusion Matrix'):  # 绘制混淆矩阵
        plt.figure(figsize=(16, 11), dpi=100)
        np.set_printoptions(precision=2)
        # 在混淆矩阵中每格的概率值
        ind_array = np.arange(len(classes) + 1)
        x, y = np.meshgrid(ind_array, ind_array)  # 生成坐标矩阵
        diags = np.diag(cm)  # 对角TP值
        TP_FNs, TP_FPs = [], []
        for x_val, y_val in zip(x.flatten(), y.flatten()):  # 并行遍历
            max_index = len(classes)
            if x_val != max_index and y_val != max_index:  # 绘制混淆矩阵各格数值
                c = cm[y_val][x_val]
                plt.text(x_val, y_val, c, color='black', fontsize=15, va='center', ha='center')
            elif x_val == max_index and y_val != max_index:  # 绘制最右列即各数据类别的查全率
                TP = diags[y_val]
                TP_FN = cm.sum(axis=1)[y_val]
                recall = TP / (TP_FN)
                if recall != 0.0 and recall > 0.01:
                    recall = str('%.2f' % (recall * 100,))
                elif recall == 0.0:
                    recall = '0'
                TP_FNs.append(TP_FN)
                plt.text(x_val, y_val, str(TP_FN) + '\n' + str(recall) + '%', color='black', va='center', ha='center')
            elif x_val != max_index and y_val == max_index:  # 绘制最下行即各数据类别的查准率
                TP = diags[x_val]
                TP_FP = cm.sum(axis=0)[x_val]
                precision = TP / (TP_FP)
                if precision != 0.0 and precision > 0.01:
                    precision = str('%.2f' % (precision * 100,)) + '%'
                elif precision == 0.0:
                    precision = '0'
                TP_FPs.append(TP_FP)
                plt.text(x_val, y_val, str(TP_FP) + '\n' + str(precision), color='black', va='center', ha='center')
        cm = np.insert(cm, max_index, TP_FNs, 1)
        cm = np.insert(cm, max_index, np.append(TP_FPs, SUM), 0)
        plt.text(max_index, max_index, str(SUM) + '\n' + str('%.2f' % (PRECISION * 100,)) + '%', color='white',
                 va='center', ha='center')
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        xlocations = np.array(range(len(classes)))
        plt.xticks(xlocations, classes, rotation=90)
        plt.yticks(xlocations, classes)
        plt.ylabel('actual label')
        plt.xlabel('predict label')
        # offset the tick
        tick_marks = np.array(range(len(classes))) + 0.5
        plt.gca().set_xticks(tick_marks, minor=True)
        plt.gca().set_yticks(tick_marks, minor=True)
        plt.gca().xaxis.set_ticks_position('none')
        plt.gca().yaxis.set_ticks_position('none')
        plt.grid(True, which='minor', linestyle='-')
        # plt.gcf().subplots_adjust(bottom=0.15)
        # show confusion matrix
        plt.savefig(savename, format='png')
        plt.show()


    plot_confusion_matrix(cm, 'cm_' + model + '.png', title='confusion matrix of ' + model)