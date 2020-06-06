from libraries import *
from itertools import combinations

import numpy as np
import torch

def unfold_label(labels, classes):
    new_labels = []

    assert len(np.unique(labels)) == classes
    # minimum value of labels
    mini = np.min(labels)

    for index in range(len(labels)):
        dump = np.full(shape=[classes], fill_value=0).astype(np.int8)
        _class = int(labels[index]) - mini
        dump[_class] = 1
        new_labels.append(dump)

    return np.array(new_labels)

def shuffle_data(samples, labels):
    num = len(labels)
    shuffle_index = np.random.permutation(np.arange(num))
    shuffled_samples = samples[shuffle_index]
    shuffled_labels = labels[shuffle_index]
    return shuffled_samples, shuffled_labels


def scoring(prob_pred):
    w = torch.from_numpy(entropy(prob_pred.cpu().detach().numpy(), base=10, axis=-1).astype(np.float32))
    score = 1-w
    return score

def cat(F_l, Y_l, F_u, Y_u):
    feat_lu = torch.cat((F_l, F_u), 0)
    Y_lu = torch.cat((Y_l, Y_u), 0)
    return feat_lu, Y_lu

def class_centroid_lu(args, F_lu, Y_lu):
    number_class = args.num_classes
    res = torch.zeros([number_class, F_lu.size(1)])
    ls_class = torch.unique(Y_lu)
    ls_class = torch.sort(ls_class)[0]
    for ite in ls_class:
        res[ite,:] = F_lu[Y_lu==ite].mean(0)
    return res.cuda()

def class_centroid(args, F, Y):
    number_class = args.num_classes
    res = torch.zeros([number_class, F.size(1)])
    ls_class = torch.unique(Y)
    ls_class = torch.sort(ls_class)[0]
    for ite in ls_class:
        res[ite,:] = F[Y==ite].mean(0)
    return res.cuda()

def row_pairwise_distances(X, y=None, dist_mat=None):
    f_mat = []
    for x in X:
        y = None
        dist_mat = None
        if y is None:
            y = x
        if dist_mat is None:
            dtype = x.data.type()
            dist_mat = Variable(torch.Tensor(x.size()[0], y.size()[0]).type(dtype))

        for i, row in enumerate(x.split(1)):
            r_v = row.expand_as(y)
            sq_dist = torch.sum((r_v - y) ** 2, 1)
            dist_mat[i] = sq_dist.view(1, -1)
        f_mat.append(dist_mat)   
    return f_mat

def global_alignment(mat1, mat2):
    dist_norm = 0
    for i in range(len(mat1)):
        for j in range(len(mat2)):
            dist_norm += torch.norm(F.normalize(mat1[i])-F.normalize(mat2[j])) 
    return dist_norm


def cat_dist_mats(args, mat1, mat2):
    number_class = args.num_classes
    cat_mat = torch.zeros([(number_class*(len(mat1)+len(mat2))), mat1[0].size(1)]).cuda()
    ind = torch.from_numpy(np.linspace(0, number_class-1, number_class).astype(np.int64)).cuda()
    for i, j in zip(mat1, mat2):
        cat_mat[ind,:] = i
        ind = ind + number_class
        cat_mat[ind,:] = j
        ind = ind + number_class
    return cat_mat




def compute_accuracy(predictions, labels):
    accuracy = accuracy_score(y_true=np.argmax(labels, axis=-1), y_pred=np.argmax(predictions, axis=-1))
    return accuracy 

def get_path(args, data_folder, train_data, val_data, test_data, unseen_index):

    train_paths = []
    for data_tr, data_ts in zip(train_data,test_data):
        path_tr = os.path.join(data_folder, data_tr)
        path_ts = os.path.join(data_folder, data_ts)
        train_paths.append([path_tr, path_ts])

    val_paths = []
    for data in val_data:
        path = os.path.join(data_folder, data)
        val_paths.append([path])  

    unseen_data_path = []

    unseen_data_path.append(train_paths[unseen_index] + val_paths[unseen_index])

    train_paths.remove(train_paths[unseen_index])
    val_paths.remove(val_paths[unseen_index])      

    return train_paths, val_paths, unseen_data_path
