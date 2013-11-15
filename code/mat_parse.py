from scipy.io import loadmat
import cPickle
import gzip
import numpy as np
import logging
FORMAT = '%(asctime)-15s: %(message)s'
logging.basicConfig(format=FORMAT, filename='../results/mat_parse.log', level=logging.DEBUG)
logger = logging.getLogger()


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
		if mx == 0:
			mx = 1
		tmp.append(map(lambda c:(c-mn)/mx, col))
		col_idx += 1
	return np.transpose(tmp)

def jumper(lst, jump=1):
	out = []
	for i in range(len(lst)):
		if i % jump == 0:
			out.append(lst[i])
	return out


def main():
	ip_fn = "../data/20Newsgroups.mat"
	logger.info("Load form the matlab dataset.")
	mat = loadmat(ip_fn)
	test_idx = mat['testIdx']
	train_idx = mat['trainIdx']
	data = mat['fea'].toarray()
	labels = mat['gnd']
	data_path = "../data/"

	logger.info("Getting min max list")
	min_max = get_min_max(data)
	logger.info("Applying Z normalization")
	data = apply_min_max(data, min_max)


	jump = 10
	valid_perc = 10
	logger.info("Getting the indices for each dataset.")
	valid_idx = map(lambda i: mat['trainIdx'][i][0],
					filter(lambda j: j % valid_perc == 0, range(len(mat['trainIdx']))))
	train_idx = map(lambda i: mat['trainIdx'][i][0],
					filter(lambda j: j % valid_perc != 0, range(len(mat['trainIdx']))))
	test_idx = [ item[0] for item in mat['testIdx'].tolist() ]

	logger.info("Reducing the datasets.")
	train_idx = jumper(train_idx, jump)
	valid_idx = jumper(valid_idx, jump)
	test_idx = jumper(test_idx, jump)

	logger.info("Getting the labels.")
	train_labels = map(lambda i: mat['gnd'][i-1][0], train_idx)
	valid_labels = map(lambda i: mat['gnd'][i-1][0], valid_idx)
	test_labels = map(lambda i: mat['gnd'][i-1][0], test_idx)

	logger.info("Getting the data.")
	train = map(lambda i: data[i-1], train_idx)
	valid = map(lambda i: data[i-1], valid_idx)
	test = map(lambda i: data[i-1], test_idx)

	set1 = (train, train_labels)
	set2 = (valid, valid_labels)
	set3 = (test, test_labels)

	logger.info("Pickling the data.")
	cPickle.dump(set1, open(data_path + "train" + "_mat.pkl", "w"))
	cPickle.dump(set2, open(data_path + "valid" + "_mat.pkl", "w"))
	cPickle.dump(set3, open(data_path + "test" + "_mat.pkl", "w"))
	cPickle.dump((set1, set2, set3), gzip.open(data_path + "Full_mat.pkl.gz", "wb"))


if __name__ == "__main__":
	try:
		main()
	except Exception as e:
		logger.exception(e)
