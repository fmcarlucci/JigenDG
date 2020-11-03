from sklearn import metrics
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
import os.path as osp
import warnings
warnings.filterwarnings('ignore')
import pickle

def acc(ypred, y, return_idx=False):
    """
    Calculating the clustering accuracy. The predicted result must have the same number of clusters as the ground truth.
    ypred: 1-D numpy vector, predicted labels
    y: 1-D numpy vector, ground truth
    The problem of finding the best permutation to calculate the clustering accuracy is a linear assignment problem.
    This function construct a N-by-N cost matrix, then pass it to scipy.optimize.linear_sum_assignment to solve the assignment problem.
    """
    assert len(y) > 0
    assert len(np.unique(ypred)) == len(np.unique(y))

    s = np.unique(ypred)
    t = np.unique(y)

    N = len(np.unique(ypred))
    C = np.zeros((N, N), dtype=np.int32)
    for i in range(N):
        for j in range(N):
            idx = np.logical_and(ypred == s[i], y == t[j])
            C[i][j] = np.count_nonzero(idx)

    # convert the C matrix to the 'true' cost
    Cmax = np.amax(C)
    C = Cmax - C
    #
    # indices = linear_sum_assignment(C)
    # row = indices[:][:, 0]
    # col = indices[:][:, 1]
    row, col = linear_sum_assignment(C)
    # calculating the accuracy according to the optimal assignment
    count = 0
    for i in range(N):
        idx = np.logical_and(ypred == s[row[i]], y == t[col[i]])
        count += np.count_nonzero(idx)

    if return_idx:
        return 1.0 * count / len(y), row, col
    else:
        return 1.0 * count / len(y)


def calculate_acc(y_pred, y_true):
    Y_pred = y_pred
    Y = y_true
    from sklearn.utils.linear_assignment_ import linear_assignment
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
        ind = linear_assignment(w.max() - w)
    return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size,ind


def calculate_nmi(predict_labels, true_labels):
    # NMI
    nmi = metrics.normalized_mutual_info_score(true_labels, predict_labels, average_method='geometric')
    return nmi


def calculate_ari(predict_labels, true_labels):
    # ARI
    ari = metrics.adjusted_rand_score(true_labels, predict_labels)
    return ari

def load_data(pkl_file):
    with open(pkl_file, 'rb') as handle:
        data = pickle.load(handle)
    return data

def cluster(data):

    kmeans = KMeans(n_clusters=7,max_iter=300, random_state=0).fit(data)
    return kmeans

def evaluate_clustering(model, data, labels):

    clusters = model.predict(data)
    acc_v, acc_i = calculate_acc(clusters, labels)
    nmi_v = calculate_nmi(clusters, labels)
    ari_v = calculate_ari(clusters, labels)
    print("Accuracy %f NMI %f ARI %f" % (acc_v, nmi_v, ari_v))
    return acc_v, nmi_v, ari_v

def shuffle(x, y):
    perm = np.random.permutation(len(y))
    x = x[perm]
    y = y[perm]
    return x,y

if __name__ == "__main__":

    root = '/content/drive/My Drive/Codes/JigenDG/logs/cartoon_target_stylizedjigsaw/art-photo-sketch_to_cartoon'
    pkl_file = 'act_labels.pkl'
    data = load_data(osp.join(root, pkl_file))
    features = data['features']
    labels = data['labels']
    features, labels = shuffle(features, labels)
    split = 0 #Use 0 to use all the data
    if split:
      count = int(len(labels)*split)
      tr_features = features[:count]
      tr_labels = labels[:count]
      te_features = features[count:]
      te_labels = labels[count:]
    else:
      tr_features = features
      te_features = features
      te_labels = labels

    model = cluster(tr_features)
    evaluate_clustering(model, te_features, te_labels)


    
    


