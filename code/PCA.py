from sklearn.decomposition import RandomizedPCA
import cPickle
from scipy.sparse import coo_matrix

from scipy import sparse
import gzip
from bisect import bisect_left

data_path = "../data/"

def jumper(lst, jump=1):
	out = []
	for i in range(len(lst)):
		if i % jump == 0:
			out.append(lst[i])
	return out

test, test_labels = cPickle.load(open("../data/test.sparse.pkl"))
train, train_labels = cPickle.load(open("../data/train.sparse.pkl"))
valid, valid_labels = cPickle.load(open("../data/validation.sparse.pkl"))

num_features = train.shape[1]

pca = RandomizedPCA(n_components=1013)

train_red = pca.fit_transform(train)
valid_red = pca.transform(valid)
test_red = pca.transform(test)

jump = 10

set1 = (jumper(test_red, jump), jumper(test_labels, jump))
set2 = (jumper(valid_red, jump), jumper(valid_labels, jump))
set3 = (jumper(train_red, jump), jumper(train_labels, jump))


try:
    cPickle.dump(set1, open(data_path + "test" + ".PCA.sparse.pkl", "w"))
    cPickle.dump(set2, open(data_path + "validation" + ".PCA.sparse.pkl", "w"))
    cPickle.dump(set3, open(data_path + "train" + ".PCA.sparse.pkl", "w"))
    cPickle.dump((set3, set2, set1), gzip.open(data_path + "PCA.sparse.pkl.gz", "wb"))
except:
    import pdb; pdb.set_trace()
