from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

classes = ['A1', 'A2', 'H1', 'L1', 'L2', 'M1', 'O1', 'S1', 'S2', 'W1', 'X1']  # 数据类别y
models = ['VISION_25-shot']  # 我的数据子路径,用于下面读取数据
annot_file = './labels/labels.txt'  # 实际的数据类别y_true
json_path = './labels/pred.txt'  # 预测的数据类别y_pred

# # 全局设置字体及大小，设置公式字体即可，若要修改刻度字体，可在此修改全局字体
# config = {
#     "mathtext.fontset":'stix',
#     "font.family":'serif',
#     "font.serif": ['SimSun'],
#     "font.size": 10,			# 字号，大家自行调节
#     'axes.unicode_minus': False # 处理负号，即-号
# }
# rcParams.update(config)

font_config = {
    "font.family": "sans-serif",
    "font.sans-serif": ["SimSun", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"],
    "axes.unicode_minus": False,
}
rcParams.update(font_config)


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

    read_file.close()

    # 计算混淆矩阵，FP，FN，TP，TN，PRECISION，RECALL
    cm = confusion_matrix(y_true, y_pred, labels=None, sample_weight=None)

    FP = sum(cm.sum(axis=0)) - sum(np.diag(cm))  # 假正样本数
    FN = sum(cm.sum(axis=1)) - sum(np.diag(cm))  # 假负样本数
    TP = sum(np.diag(cm))  # 真正样本数
    TN = sum(cm.sum().flatten()) - (FP + FN + TP)  # 真负样本数
    SUM = TP + FP
    PRECISION = TP / (TP + FP)  # 查准率，又名准确率
    print(PRECISION)
    RECALL = TP / (TP + FN)  # 查全率，又名召回率


    def plot_confusion_matrix(cm, savename, title='Confusion Matrix'):  # 绘制混淆矩阵
        plt.figure(figsize=(12, 10), dpi=100)
        np.set_printoptions(precision=2)
        # 在混淆矩阵中每格的概率值
        ind_array = np.arange(len(classes) + 1)
        x, y = np.meshgrid(ind_array, ind_array)  # 生成坐标矩阵
        diags = np.diag(cm)  # 对角TP值
        TP_FNs, TP_FPs = [], []
        recall_cm = np.zeros((len(classes), len(classes)))
        for x_val, y_val in zip(x.flatten(), y.flatten()):  # 并行遍历
            max_index = len(classes)
            if x_val != max_index and y_val != max_index:  # 绘制混淆矩阵各格数值
                c = cm[y_val][x_val]
                TP_FN = cm.sum(axis=1)[y_val]
                recall_every = c / (TP_FN)
                if recall_every != 0.0 and recall_every >= 0.005:
                    recall_every = str('%.2f' % (recall_every * 100,))
                elif recall_every < 0.005:
                    recall_every = '0.00'
                recall_cm[y_val][x_val] = recall_every
                if y_val == x_val:
                    plt.text(x_val, y_val, recall_every, color='white', fontsize=15, va='center', ha='center')
                else:
                    plt.text(x_val, y_val, recall_every, color='black', fontsize=15, va='center', ha='center')
            elif x_val == max_index and y_val != max_index:  # 绘制最右列即各数据类别的查全率
                TP = diags[y_val]
                TP_FN = cm.sum(axis=1)[y_val]
                recall = TP / (TP_FN)
                if recall != 0.0 and recall > 0.01:
                    recall = str('%.2f' % (recall * 100,))
                elif recall == 0.0:
                    recall = '0'
                TP_FNs.append(TP_FN)
                # plt.text(x_val, y_val, str(TP_FN) + '\n' + str(recall) + '%', color='black', va='center', ha='center')
            elif x_val != max_index and y_val == max_index:  # 绘制最下行即各数据类别的查准率
                TP = diags[x_val]
                TP_FP = cm.sum(axis=0)[x_val]
                precision = TP / (TP_FP)
                if precision != 0.0 and precision > 0.01:
                    precision = str('%.2f' % (precision * 100,)) + '%'
                elif precision == 0.0:
                    precision = '0'
                TP_FPs.append(TP_FP)
                # plt.text(x_val, y_val, str(TP_FP) + '\n' + str(precision), color='black', va='center', ha='center')
        # cm = np.insert(cm, max_index, TP_FNs, 1)
        # cm = np.insert(cm, max_index, np.append(TP_FPs, SUM), 0)
        # 绘制右下角整体准确率
        # plt.text(max_index, max_index, str(SUM) + '\n' + str('%.2f' % (PRECISION * 100,)) + '%', color='white',
        #          va='center', ha='center')
        plt.imshow(recall_cm, interpolation='nearest', cmap=plt.cm.Blues)
        # # plt.title(title)
        plt.colorbar()
        xlocations = np.array(range(len(classes)))
        plt.xticks(xlocations, classes, rotation=45)
        plt.yticks(xlocations, classes)
        plt.ylabel(u'真实标签', fontsize=15)
        plt.xlabel(u'预测标签', fontsize=15)
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
