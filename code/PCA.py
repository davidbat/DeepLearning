from sklearn.decomposition import RandomizedPCA, ProbabilisticPCA, NMF
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

def dec_labels_by_one(lst):
	return map(lambda r:r-1, lst)

def find_min(lst):
	return abs(min(map(lambda m: min(m), lst)))

def add_min(lst, global_min):
	return map(lambda row: map(lambda itm: itm+global_min, row), lst)

test, test_labels = cPickle.load(open("../data/test.sparse.pkl"))
train, train_labels = cPickle.load(open("../data/train.sparse.pkl"))
valid, valid_labels = cPickle.load(open("../data/validation.sparse.pkl"))

num_features = train.shape[1]
jump = 10

## The min stuff really slows down code... But it is necessary for positive costs
## TODO: Find a more efficient way to get positive values after 
train = train.toarray()
test = test.toarray()
valid = valid.toarray()


test_labels = dec_labels_by_one(test_labels)
train_labels = dec_labels_by_one(train_labels)
valid_labels = dec_labels_by_one(valid_labels)

pca = NMF(n_components=1013)

train_red = pca.fit_transform(train)
valid_red = pca.transform(valid)
test_red = pca.transform(test)

train_red = jumper(train_red, jump)
valid_red = jumper(valid_red, jump)
test_red = jumper(test_red, jump)

# mins = []
# mins.append(find_min(train_red))
# mins.append(find_min(test_red))
# mins.append(find_min(valid_red))

# global_min = min(mins)
# train_red = add_min(train_red, global_min)
# test_red = add_min(test_red, global_min)
# valid_red = add_min(valid_red, global_min)


set1 = (train_red, jumper(train_labels, jump))
set2 = (valid_red, jumper(valid_labels, jump))
set3 = (test_red, jumper(test_labels, jump))


try:
    cPickle.dump(set1, open(data_path + "train" + ".PCA.sparse.pkl", "w"))
    cPickle.dump(set2, open(data_path + "validation" + ".PCA.sparse.pkl", "w"))
    cPickle.dump(set3, open(data_path + "test" + ".PCA.sparse.pkl", "w"))
    cPickle.dump((set1, set2, set3), gzip.open(data_path + "PCA.sparse.pkl.gz", "wb"))
except:
    import pdb; pdb.set_trace()
