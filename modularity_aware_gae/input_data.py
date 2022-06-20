import networkx as nx
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import sys


"""
Disclaimer: the parse_index_file function from this file, as well as the
cora/citeseer/pubmed parts of the loading functions, come from the
tkipf/gae original repository on Graph Autoencoders
"""


def parse_index_file(filename):

    index = []
    for line in open(filename):
        index.append(int(line.strip()))

    return index


def load_data(dataset):

    """
    Load datasets
    :param dataset: name of the input graph dataset
    :return: n*n sparse adjacency matrix and n*f node features matrix
    """

    if dataset == 'cora-large':
        adj = nx.adjacency_matrix(nx.read_edgelist("../data/coralarge", delimiter = ' '))
        features = sp.identity(adj.shape[0])

    elif dataset == 'sbm':
        adj = nx.adjacency_matrix(nx.read_edgelist("../data/sbm.txt"))
        features = sp.identity(adj.shape[0])

    elif dataset == 'blogs':
        adj = nx.adjacency_matrix(nx.read_edgelist("../data/blogs",
                                                   nodetype = int,
                                                   data = (('weight', int),),
                                                   delimiter = ' '))
        features = sp.identity(adj.shape[0])

    elif dataset in ('cora', 'citeseer', 'pubmed'):
        # Load the data: x, tx, allx, graph
        names = ['x', 'tx', 'allx', 'graph']
        objects = []
        for i in range(len(names)):
            with open("../data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding = 'latin1'))
                else:
                    objects.append(pkl.load(f))
        x, tx, allx, graph = tuple(objects)
        test_idx_reorder = parse_index_file("../data/ind.{}.test.index".format(dataset))
        test_idx_range = np.sort(test_idx_reorder)
        if dataset == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range - min(test_idx_range), :] = tx
            tx = tx_extended
        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        graph = nx.from_dict_of_lists(graph)
        adj = nx.adjacency_matrix(graph)

    else:
        raise ValueError('Error: undefined dataset!')

    return adj, features


def load_labels(dataset):

    """
    Load node-level labels
    :param dataset: name of the input graph dataset
    :return: n-dim array of node labels, used for community detection
    """

    if dataset == 'cora-large':
        labels = np.loadtxt("../data/coralarge-cluster", delimiter = ' ', dtype = str)

    elif dataset == 'sbm':
        labels = np.repeat(range(100), 1000)

    elif dataset == 'blogs':
        labels = np.loadtxt("../data/blogs-cluster", delimiter = ' ', dtype = str)

    elif dataset in ('cora', 'citeseer', 'pubmed'):
        names = ['ty', 'ally']
        objects = []
        for i in range(len(names)):
            with open("../data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding = 'latin1'))
                else:
                    objects.append(pkl.load(f))
        ty, ally = tuple(objects)
        test_idx_reorder = parse_index_file("../data/ind.{}.test.index".format(dataset))
        test_idx_range = np.sort(test_idx_reorder)
        if dataset == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
            ty_extended = np.zeros((len(test_idx_range_full), ty.shape[1]))
            ty_extended[test_idx_range - min(test_idx_range), :] = ty
            ty = ty_extended
        labels = sp.vstack((ally, ty)).tolil()
        labels[test_idx_reorder, :] = labels[test_idx_range, :]
        # One-hot to integers
        labels = np.argmax(labels.toarray(), axis = 1)

    else:
        raise ValueError('Error: undefined dataset!')

    return labels