from sklearn.decomposition import RandomizedPCA
import cPickle
from scipy.sparse import coo_matrix

from scipy import sparse
import gzip
from bisect import bisect_left

data_path = "../data/"

class lil2(sparse.lil_matrix):
    def removecol(self,j):
        if j < 0:
            j += self.shape[1]

        if j < 0 or j >= self.shape[1]:
            raise IndexError('column index out of bounds')

        rows = self.rows
        data = self.data
        for i in xrange(self.shape[0]):
            pos = bisect_left(rows[i], j)
            if pos == len(rows[i]):
                continue
            elif rows[i][pos] == j:
                rows[i].pop(pos)
                data[i].pop(pos)
                if pos == len(rows[i]):
                    continue
            for pos2 in xrange(pos,len(rows[i])):
                rows[i][pos2] -= 1

        self._shape = (self._shape[0],self._shape[1]-1)


test, test_labels = cPickle.load(open("../data/test.sparse.pkl"))
train, train_labels = cPickle.load(open("../data/train.sparse.pkl"))
valid, valid_labels = cPickle.load(open("../data/validation.sparse.pkl"))

num_features = train.shape[1]

pca = RandomizedPCA(n_components=2000)

# lil_train = train.tolil()
# train_new = lil2(lil_train)

pca.fit(train)


train = pca.transform(train)
test = pca.transform(test)
valid = pca.transform(valid)

set1 = (test, test_labels)
set2 = (valid, valid_labels)
set3 = (train, train_labels)

try:
    cPickle.dump(set1, open(data_path + "test" + ".PCA.sparse.pkl", "w"))
    cPickle.dump(set2, open(data_path + "validation" + ".PCA.sparse.pkl", "w"))
    cPickle.dump(set3, open(data_path + "train" + ".PCA.sparse.pkl", "w"))
    cPickle.dump((set3, set2, set1), gzip.open(data_path + "PCA.sparse.pkl", "wb"))
except:
    import pdb; pdb.set_trace()
