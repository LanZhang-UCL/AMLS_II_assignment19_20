import math
from pandas import read_csv
from sklearn.metrics import precision_recall_fscore_support


# ======================================================================================================================
# Split results based on topic.
def split_topic(topic, y_true, y_pred):
    new_topic = []
    new_true = []
    new_pred = []
    for i in range(0, len(topic)):
        if topic[i] in new_topic:
            j = new_topic.index(topic[i])
            new_true[j].append(y_true[i])
            new_pred[j].append(y_pred[i])
        else:
            new_topic.append(topic[i])
            new_true.append([y_true[i]])
            new_pred.append([y_pred[i]])
    return new_topic, new_true, new_pred


# ======================================================================================================================
# Further tests on B's prediction
B_df = read_csv('./B.csv')
topic = B_df['topic']
y_true_B = B_df['label']
y_pred_B = B_df['predict_label']

topic, y_true_B, y_pred_B = split_topic(topic, y_true_B, y_pred_B)


# Macro-averaged precision, recall and F1-score, averaged across all topics
def average_precision_recall_fscore(topic, y_true, y_pred):
    pre = 0
    rec = 0
    F1 = 0

    # sum
    for i in range(0, len(topic)):
        temp1, temp2, temp3, _ = precision_recall_fscore_support(y_true[i], y_pred[i], average='macro')
        pre += temp1
        rec += temp2
        F1 += temp3

    # average
    return pre/len(topic), rec/len(topic), F1/len(topic)


pre_B, rec_B, F1_B = average_precision_recall_fscore(topic, y_true_B, y_pred_B)
print('B-precision:{}, B-recall:{}, B-F1score:{}'.format(pre_B, rec_B, F1_B))


# ======================================================================================================================
# Further tests on D's prediction
D_df = read_csv('./D.csv')
topic = D_df['topic']
y_true_D = D_df['label']
y_pred_D = D_df['predict_label']
epsilon = 1.0/(2*len(topic))

topic, y_true_D, y_pred_D = split_topic(topic, y_true_D, y_pred_D)


# Kullback-Leibler Divergence (KLD) along with additive smoothing, averaged across all topics
def average_KLD(topic, y_true, y_pred, epsilon):
    KLD = 0
    for i in range(0, len(topic)):
        p = [0, 0]
        q = [0, 0]
        # count
        for j in range(0, len(y_true[i])):
            p[y_true[i][j]] = p[y_true[i][j]] + 1
            q[y_pred[i][j]] = q[y_pred[i][j]] + 1

        # probability
        for j in range(0, len(p)):
            p[j] = p[j] /len(y_true[i])
            q[j] = q[j] /len(y_pred[i])

        # additive smoothing
        for j in range(0, len(p)):
            p[j] = (p[j] + epsilon)/(1 + epsilon*len(y_true[i]))
            q[j] = (q[j] + epsilon)/(1 + epsilon*len(y_pred[i]))

        # sum
        for j in range(0, len(p)):
            KLD = KLD + p[j]*math.log(p[j]/q[j])

    # average
    return KLD/len(topic)


KLD_D = average_KLD(topic, y_true_D, y_pred_D, epsilon)
print('D-KLD:{}'.format(KLD_D))