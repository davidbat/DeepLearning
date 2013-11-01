from sklearn.decomposition import RandomizedPCA, ProbabilisticPCA, NMF
import cPickle
from scipy.sparse import coo_matrix

from scipy import sparse
import gzip
from bisect import bisect_left
import numpy as np
import logging
data_path = "../data/"

FORMAT = '%(asctime)-15s: %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger()
logger.setLevel(logging.getLevelName('INFO'))
#logger.warning('Protocol problem: %s', 'connection reset', extra=d)

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

def get_min_max(arr):
	trp = np.transpose(arr)
	tmp = []
	for col in trp:
		mn = min(col)
		mx = max(col) - mn
		tmp.append((mn, mx))
	return tmp

def apply_min_max(arr, min_max):
	trp = np.transpose(arr)
	tmp = []
	col_idx = 0
	for col in trp:
		mn = min_max[col_idx][0]
		mx = min_max[col_idx][1]
		tmp.append(map(lambda c:(c-mn)/mx, col))
		col_idx += 1
	return tmp


logger.info("extracting data...\n")
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

logger.info("decreasing label value by 1...\n")
test_labels = dec_labels_by_one(test_labels)
train_labels = dec_labels_by_one(train_labels)
valid_labels = dec_labels_by_one(valid_labels)

pca = NMF(n_components=1013)

logger.info("running the transform...\n")
train_red = pca.fit_transform(train)
valid_red = pca.transform(valid)
test_red = pca.transform(test)

train_red = jumper(train_red, jump)
valid_red = jumper(valid_red, jump)
test_red = jumper(test_red, jump)

logger.info("ceating full array...\n")
full_arr = np.concatenate((train_red, valid_red, test_red), axis=0)
logger.info("get min_max...\n")
min_max = get_min_max(full_arr)
logger.info("apply min_max...\n")
train_red = apply_min_max(train_red, min_max)
valid_red = apply_min_max(valid_red, min_max)
test_red = apply_min_max(test_red, min_max)

# mins = []
# mins.append(find_min(train_red))
# mins.append(find_min(test_red))
# mins.append(find_min(valid_red))

# global_min = min(mins)
# train_red = add_min(train_red, global_min)
# test_red = add_min(test_red, global_min)
# valid_red = add_min(valid_red, global_min)

logger.info("creating sets...\n")
set1 = (train_red, jumper(train_labels, jump))
set2 = (valid_red, jumper(valid_labels, jump))
set3 = (test_red, jumper(test_labels, jump))


try:
    logger.info("pickling the data...\n")
    cPickle.dump(set1, open(data_path + "train" + ".PCA.sparse.pkl", "w"))
    cPickle.dump(set2, open(data_path + "validation" + ".PCA.sparse.pkl", "w"))
    cPickle.dump(set3, open(data_path + "test" + ".PCA.sparse.pkl", "w"))
    cPickle.dump((set1, set2, set3), gzip.open(data_path + "PCA.sparse.pkl.gz", "wb"))
except:
    logger.info("error while pickling...\n")
    #import pdb; pdb.set_trace()
