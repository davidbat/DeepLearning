from sklearn.decomposition import RandomizedPCA
import cPickle
from scipy.sparse import coo_matrix

from scipy import sparse
import gzip
from bisect import bisect_left

data_path = "../data/"


test, test_labels = cPickle.load(open("../data/test.sparse.pkl"))
train, train_labels = cPickle.load(open("../data/train.sparse.pkl"))
valid, valid_labels = cPickle.load(open("../data/validation.sparse.pkl"))

num_features = train.shape[1]

import pdb; pdb.set_trace()
pca = RandomizedPCA(n_components=2000)

train_red = pca.fit_transform(train)
valid_red = pca.transform(valid)
test_red = pca.transform(test)

set1 = (test_red, test_labels)
set2 = (valid_red, valid_labels)
set3 = (train_red, train_labels)

try:
    cPickle.dump(set1, open(data_path + "test" + ".PCA.sparse.pkl", "w"))
    cPickle.dump(set2, open(data_path + "validation" + ".PCA.sparse.pkl", "w"))
    cPickle.dump(set3, open(data_path + "train" + ".PCA.sparse.pkl", "w"))
    cPickle.dump((set3, set2, set1), gzip.open(data_path + "PCA.sparse.pkl.gz", "wb"))
except:
    import pdb; pdb.set_trace()
