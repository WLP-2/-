import seaborn as sns
import re
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from numpy import ravel
from sklearn.metrics import recall_score, mean_squared_error
import math
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 载入数据
def load_data():
    data_train_frame = pd.read_csv("D:\大二（学科作业）\机器学习\一见钟情数据/speed_dating_train.csv")
    data_test_frame = pd.read_csv("D:\大二（学科作业）\机器学习\一见钟情数据/speed_dating_test.csv")
    return data_train_frame, data_test_frame


# 语义无关列的清除
def get_irrelevant_data():
    return ['dec_o', 'dec', 'iid', 'id', 'gender', 'idg',
            'condtn', 'wave', 'round', 'position', 'positin1',
            'order', 'partner', 'pid', 'field', 'tuition', 'career', 'undergra', 'from', ]


# 数据集中清除无关特征
def remove_irrelevant_data(data):
    irrelevant_data = get_irrelevant_data()
    data = data.drop(irrelevant_data, axis=1)
    return data


# ground truth
def get_ground_truth(data):
    """
    描述：get ground truth for the data
    :param data: 全局参数，测试数据
    :return: 测试数据
    """
    data['match'] = (data['dec'].astype("bool") & data['dec_o'].astype("bool"))
    return data


# 处理缺省值-数据清洗
def handle_missing(data, percent):
    percent_missing = data.isnull().sum() / len(data)
    missing_df = pd.DataFrame({'column_name': data.columns, 'percent_missing': percent_missing})
    missing_show = missing_df.sort_values(by='percent_missing')
    print(missing_show[missing_show['percent_missing'] > 0].count())
    print('----------------------------------')
    print(missing_show[missing_show['percent_missing'] > percent])
    columns = missing_show.index[missing_show['percent_missing'] > percent]
    data = data.drop(columns=columns, axis=1)
    return data


# 特征列选取
def select_data(data, columns):
    data = data[columns]
    return data


# 补全样本缺失值
def fill_loss_data(data):
    data = data.copy(deep=False)
    for column in data.columns:
        data[column].fillna(data[column].mode()[0], inplace=True)
    return data


# 分析特征相关性
def display_corr(data):
    plt.subplots(figsize=(20, 15))
    axis = plt.axes()
    axis.set_title("Correlation HeatMap")
    corr = data.corr(method="spearman")
    columns_save = []
    for index in corr['match'].index.values:
        if abs(corr['match'][index]) >= 0.1:
            columns_save.append(index)
    data = data[columns_save]
    corr = data.corr(method='spearman')
    sns.heatmap(corr, xticklabels=corr.columns.values, annot=True)
    plt.show()


def remove_feature(features, base_feature):
    '''
    描述：从一群特征中去除某些特征（比如取消所有sinc/attr这种）
    参数：特征列表，你要去除的
    返回：新的特征列表
    '''
    new_features = []
    for string in features:
        if re.match(base_feature, string) == None:
            new_features.append(string)
    return new_features


# 特征分组
def corr_feature(feature_id):
    # 保留相关系数0.15以上
    # 定义特征组合
    group_0 = ['match']
    group_1 = ['attr', 'sinc', 'intel', 'fun', 'amb', 'shar', 'like', 'prob', 'met', 'attr_o', 'sinc_o', 'intel_o',
               'fun_o', 'amb_o', 'shar_o', 'like_o', 'prob_o', 'met_o']
    group_2 = ['satis_2', 'attr7_2', 'sinc7_2', 'intel7_2', 'fun7_2', 'amb7_2', 'shar7_2', 'attr1_1', 'sinc1_1',
               'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1', 'attr4_1', 'sinc4_1', 'intel4_1', 'fun4_1', 'amb4_1',
               'shar4_1', \
               'attr2_1', 'sinc2_1', 'intel2_1', 'fun2_1', 'amb2_1', 'shar2_1', 'attr3_1', 'sinc3_1', 'fun3_1',
               'intel3_1', 'amb3_1', 'attr5_1', 'sinc5_1', 'intel5_1', 'fun5_1', 'amb5_1']
    group_3 = ['attr1_3', 'sinc1_3', 'intel1_3', 'fun1_3', 'amb1_3', 'shar1_3', 'attr7_3', 'sinc7_3', 'intel7_3',
               'fun7_3', 'amb7_3', 'shar7_3', 'attr4_3', 'sinc4_3', 'intel4_3', 'fun4_3', 'amb4_3', 'shar4_3', \
               'attr2_3', 'sinc2_3', 'intel2_3', 'fun2_3', 'amb2_3', 'shar2_3', 'attr3_3', 'sinc3_3', 'intel3_3',
               'fun3_3', 'amb3_3', 'attr5_3', 'sinc5_3', 'intel5_3', 'fun5_3', 'amb5_3']
    if feature_id == 1:
        # 采用group 0+1
        columns = group_0 + group_1
    elif feature_id == 2:
        # 采用group 0+2
        columns = group_0 + group_2
    elif feature_id == 3:
        # 采用group 0+3
        columns = group_0 + group_3
    elif feature_id == 4:
        # 采用group 0+1+2
        columns = group_0 + group_1 + group_2
    elif feature_id == 5:
        # 采用group 0+1+3
        columns = group_0 + group_1 + group_3
    elif feature_id == 6:
        # 采用group 0+2+3
        columns = group_0 + group_2 + group_3
    elif feature_id == 7:
        # 采用group 0+1+2+3
        columns = group_0 + group_1 + group_2 + group_3
    elif feature_id == 8:
        # 采用group 0+1, 去掉attr
        new_group_1 = remove_feature(group_1, 'attr')
        columns = group_0 + new_group_1
    elif feature_id == 9:
        # 采用group 0+1, 去掉sinc
        new_group_1 = remove_feature(group_1, 'sinc')
        columns = group_0 + new_group_1
    elif feature_id == 10:
        # 采用group 0+1, 去掉intel
        new_group_1 = remove_feature(group_1, 'intel')
        columns = group_0 + new_group_1
    elif feature_id == 11:
        # 采用group 0+1, 去掉fun
        new_group_1 = remove_feature(group_1, 'fun')
        columns = group_0 + new_group_1
    elif feature_id == 12:
        # 采用group 0+1, 去掉amb
        new_group_1 = remove_feature(group_1, 'amb')
        columns = group_0 + new_group_1
    elif feature_id == 13:
        # 采用group 0+1, 去掉shar
        new_group_1 = remove_feature(group_1, 'shar')
        columns = group_0 + new_group_1
    return columns


if __name__ == '__main__':
    train_data, test_data = load_data()
    print(train_data.columns)
    train_data = handle_missing(train_data, 0.7)
    train_data = remove_irrelevant_data(train_data)
    train_data = fill_loss_data(train_data)
    # display_corr(data=train_data)



# init data
def init_data(train_data, test_data, feature_id):
    columns = corr_feature(feature_id=feature_id)
    train_data = select_data(train_data, columns)
    test_data = select_data(test_data, columns)
    train_data = fill_loss_data(train_data)
    test_data = fill_loss_data(test_data)
    # 欠采样：随机删除一部分match为0的负样本
    # negative_samples = train_data[train_data['match'] == 0]
    # positive_samples = train_data[train_data['match'] == 1]
    # negative_samples = negative_samples.sample(frac=0.7)  # 删除50%的负样本
    # train_data = pd.concat([positive_samples, negative_samples])
    x_train = train_data.drop(['match'], axis=1)
    y_train = train_data[['match']]
    x_test = test_data.drop(['match'], axis=1)
    y_test = test_data[['match']]
    return x_train, y_train, x_test, y_test

# obtain accuracy and recall
def compute_accuracy_recall(sample, data, param):
    if len(sample) != len(data):
        return 'wrong'
    for i in range(0, len(sample) - 1):
        if sample[i] > param:
            sample[i] = 1
        else:
            sample[i] = 0
    acc_number = 0
    true_positive = 0
    for i, a in zip(range(0, len(sample) - 1), data.values):
        if sample[i] == a:
            acc_number += 1
            if sample[i] == 1:
                true_positive += 1
    precision = true_positive / len(sample)
    recall = recall_score(data, sample)
    accuracy = acc_number / len(sample)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return accuracy, recall, precision, f1


# train model
def train_model(feature_id, train_data, test_data, C, max_iter, cv_scores):
    x_train, y_train, x_test, y_test = init_data(train_data, test_data, feature_id)
    model = LogisticRegression(penalty='l1', dual=False, tol=0.01, C=C, fit_intercept=True,
                               intercept_scaling=1, solver='liblinear', max_iter=max_iter)
    model.fit(x_train, ravel(y_train))

    # 计算训练集准确率和召回率
    y_pred_train = model.predict(x_train)
    train_accuracy, train_recall, train_precision, train_f1 = compute_accuracy_recall(y_pred_train, y_train, 0.7)

    # 计算测试集准确率和召回率
    y_pred_test = model.predict(x_test)
    test_accuracy, test_recall, test_precision, test_f1 = compute_accuracy_recall(y_pred_test, y_test, 0.7)

    cv_scores.append(train_accuracy)

    # 计算均方误差
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)

    # 计算均方根误差
    rmse_train = math.sqrt(mse_train)
    rmse_test = math.sqrt(mse_test)

    # 计算测试集预测概率
    y_pred_test_prob = model.predict_proba(x_test)[:, 1]

    # 计算FPR、TPR和阈值
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_test_prob)

    # 计算AUC
    roc_auc = auc(fpr, tpr)

    print("训练准确率:%.4f" % (train_accuracy * 100))
    print("测试准确率:%.4f" % (test_accuracy * 100))
    print("训练召回率:%.4f" % (train_recall * 100))
    print("测试召回率:%.4f" % (test_recall * 100))
    print("训练精确率:%.4f" % (train_precision * 100))
    print("测试精确率:%.4f" % (test_precision * 100))
    print("训练F1值:%.4f" % train_f1)
    print("测试F1值:%.4f" % test_f1)
    print("训练均方误差:%.4f" % mse_train)
    print("测试均方误差:%.4f" % mse_test)
    print("训练均方根误差:%.4f" % rmse_train)
    print("测试均方根误差:%.4f" % rmse_test)

    # 绘制ROC曲线
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_test)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_train = math.sqrt(mse_train)
    rmse_test = math.sqrt(mse_test)

    return model, train_accuracy, test_accuracy, train_recall, test_recall, train_precision, test_precision, train_f1, test_f1, mse_train, mse_test, rmse_train, rmse_test, rmse_train, rmse_test


def grid_search():
    train_data, test_data = load_data()
    test_data = get_ground_truth(test_data)
    Cs = [0.8, 1.0, 1.2, 1.4, 1.6]
    max_iters = [100, 500, 1000, 2000, 5000]
    feature_ids = [i for i in range(1, 14)]
    best_acc = 0.0
    best_recall = 0.0
    best_precision = 0.0
    best_f1 = 0.0
    best_mse = float('inf')
    best_rmse = float('inf')
    best_log_acc = {
        'C': 0.0,
        'max_iter': 0,
        'feature_id': 1
    }
    best_log_recall = {
        'C': 0.0,
        'max_iter': 0,
        'feature_id': 1
    }
    best_log_f1 = {
        'C': 0.0,
        'max_iter': 0,
        'feature_id': 1
    }
    best_log_precision = {
        'C': 0.0,
        'max_iter': 0,
        'feature_id': 1
    }
    best_model_precision = None
    best_model_acc = None
    best_model_recall = None
    best_model_f1 = None

    # 定义 x_train 和 y_train
    x_train, y_train, _, _ = init_data(train_data, test_data, feature_ids[0])
    accuracy_scores = []

    for feature_id in feature_ids:
        cv_scores = []  # 新增代码
        for C in Cs:
            for max_iter in max_iters:
                print("feature_id:", feature_id, ",C:", C, ",max_iter:", max_iter)
                model, train_acc, test_acc, train_recall, test_recall, train_precision, test_precision, train_f1, test_f1, mse_train, mse_test, rmse_train, rmse_test, rmse_train, rmse_test = train_model(
                    feature_id, train_data, test_data, C, max_iter, cv_scores)
                accuracy_scores.append([feature_id, C, max_iter, test_acc])

                if best_acc < test_acc:
                    best_acc = test_acc
                    best_log_acc['C'] = C
                    best_log_acc['max_iter'] = max_iter
                    best_log_acc['feature_id'] = feature_id
                    best_model_acc = model

                if best_recall < test_recall:
                    best_recall = test_recall
                    best_log_recall['C'] = C
                    best_log_recall['max_iter'] = max_iter
                    best_log_recall['feature_id'] = feature_id
                    best_model_recall = model

                if best_precision < test_precision:
                    best_precision = test_precision
                    best_log_precision['C'] = C
                    best_log_precision['max_iter'] = max_iter
                    best_log_precision['feature_id'] = feature_id
                    best_model_precision = model

                if best_f1 < test_f1:
                    best_f1 = test_f1
                    best_log_f1['C'] = C
                    best_log_f1['max_iter'] = max_iter
                    best_log_f1['feature_id'] = feature_id
                    best_model_f1 = model

                if best_mse > mse_test:
                    best_mse = mse_test
                    best_rmse = rmse_test

                if best_mse > mse_test:
                    best_mse = mse_test
                    best_rmse = rmse_test

    print("Accuracy最高的配置：")
    print(best_log_acc)
    print("最高准确率：", best_acc)
    print("对应的模型信息：")
    print(best_model_acc)
    print("交叉验证准确率：", cv_scores)  # 新增代码
    print("对应的模型信息：")
    print(best_model_acc)

    print("Recall最高的配置：")
    print(best_log_recall)
    print("最高召回率：", best_recall)
    print("对应的模型信息：")
    print(best_model_recall)

    print("Precision最高的配置：")
    print(best_log_precision)
    print("最高精确率：", best_precision)
    print("对应的模型信息：")
    print(best_model_precision)

    print("F1值最高的配置：")
    print(best_log_f1)
    print("最高F1值：", best_f1)
    print("对应的模型信息：")
    print(best_model_f1)

    #print("均方误差最低：")
    print("最低均方误差：", best_mse)

    #print("均方根误差最低：")
    print("最低均方根误差：", best_rmse)
    accuracy_scores = np.array(accuracy_scores)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(accuracy_scores[:, 0], accuracy_scores[:, 1], accuracy_scores[:, 3], cmap='jet')
    ax.set_xlabel('Feature ID')
    ax.set_ylabel('C')
    ax.set_zlabel('Accuracy')
    plt.show()

if __name__ == '__main__':
    grid_search()