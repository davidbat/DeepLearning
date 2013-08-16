import sys
from optparse import OptionParser
import cPickle
from scipy.sparse import coo_matrix
import numpy as np
# cPickle.dump(myobj, open("myfile.pickle", "w"))
# myobj2 = cPickle.load(open("myfile.pickle"))

data_path = "../data/"

training_set = "train"

testing_set = "test"

stop_ids = "stop_word_ids.txt"
features = "vocabulary.txt"
validation_perc = 10


def readFile(fn):
  return [ line.strip() for line in open(data_path + fn).readlines() ]

def readCol(fn, stop_list, delim=" "):
	tmp_lst = [ line.strip().split(delim) for line in open(data_path + fn).readlines() ]
	return filter(lambda row: row[1] not in stop_list, tmp_lst)

def indexedHasher(fn):
	lines = [ line.strip() for line in open(data_path + fn).readlines() ]
	return {key: value for (key, value) in map(lambda lk, lv: (lk, lv), range(1, len(lines) + 1),  lines)}

def calculateSparseDict(data_set, data_label, jump=1):
	docs = {}
	for i in range(len(data_set)):
		try:
			if int(data_set[i][0]) not in docs:
				if not int(data_set[i][0]) % jump == 0:
					continue
				docs[int(data_set[i][0])] = { int(data_set[i][1]): int(data_set[i][2]), 'label': data_label[int(data_set[i][0])]}
			else:
				docs[int(data_set[i][0])][int(data_set[i][1])] = int(data_set[i][2])
		except:
			print "Exception"
			sys.exit(1)
	return docs

def translate(data):
	########  REALLY SLOW CODE!!!! OPTIMIZE!!!  ########
	"""input must be a sorted list of positive integers"""
	"""make the given array a continous set of numbers
	   ex - translate([1,1,1,33,33,55,55]) -> [1,1,1,2,2,3,3] """
	out = []
	# lens = [data.count(k) for k in sorted(set(data))]
	# nested = map(lambda indx:[indx for i in range(lens[indx-1])],range(1, len(lens)+1))
	# return [item for sublist in nested for item in sublist]
	hsh = {j: data.count(j) for j in set(data)}
	cnt = 0
	for k in sorted(hsh.keys()):
		out += [cnt for i in range(hsh[k])]
		cnt += 1
	# ctr = 0
	# prev = data[0]
	# for point in data:
	# 	if point != prev:
	# 		prev = point
	# 		ctr += 1
	# 	out.append(ctr)
	return out

def calculateSparseDictCOO(data_set, data_label_hash, jump=1, valid_flag=False):
	row = []
	col = []
	data = []
	row_valid = []
	col_valid = []
	data_valid = []

	doc_ids = set(sorted(map(lambda row:int(row[0]), data_set)))
	base_ids = set(filter(lambda ids: ids % jump == 0, doc_ids))
	train_ids = base_ids
	valid_ids = set()
	if valid_flag:
		valid_ids = set(filter(lambda ids: ids % validation_perc == 0, base_ids))
		train_ids = sorted(base_ids - valid_ids)
		valid_ids = sorted(valid_ids)

	labels = map(lambda trid: int(data_label_hash[trid]), train_ids)
	labels_valid = map(lambda vlid: int(data_label_hash[vlid]), valid_ids)
	for i in range(len(data_set)):
		if int(data_set[i][0]) in train_ids:
			row.append(int(data_set[i][0]))
			col.append(int(data_set[i][1])-1)
			data.append(int(data_set[i][2]))
			# labels.append(int(data_label_hash[int(data_set[i][0])]))
		elif int(data_set[i][0]) in valid_ids:
			row_valid.append(int(data_set[i][0]))
			col_valid.append(int(data_set[i][1])-1)
			data_valid.append(int(data_set[i][2]))
			# labels_valid.append(int(data_label_hash[int(data_set[i][0])]))

	train = translate(row), col, data, labels
	valid = translate(row_valid), col_valid, data_valid, labels_valid
	return train, valid

def writeOneList(doc, label, fn):
	fd = open(fn, 'a')
	fd.write(", ".join(doc) + ", " + label + "\n")
	fd.close()

def calculateFullList(data_set, data_label, features_num, fn, jump=1):
	doc = []
	prev_id = -1
	for i in range(len(data_set)):
		if not data_set[i][0] == prev_id:
			if not (int(data_set[i][0]) % jump == 0):
				continue
			if doc:
				writeOneList(doc, data_label[int(prev_id)],  fn)
			prev_id = data_set[i][0]
			doc = [ '0' for j in range(features_num) ]
			doc[int(data_set[i][1]) - 1] = data_set[i][2]
			# import pdb; pdb.set_trace()
			# doc[features_num] = data_label[int(data_set[i][0])]
			# tmp_lst = [ '0' for i in range(features_num + 1) ]
			# docs[data_set[i][0]] = tmp_lst
			# docs[data_set[i][0]][int(data_set[i][1])] = data_set[i][2]
			# docs[data_set[i][0]][features_num] = data_label[int(data_set[i][0])]
		else:
			try:
				doc[int(data_set[i][1]) - 1] = data_set[i][2]
			except:
				print "An error has occured and we have entered debug mode."
				import pdb; pdb.set_trace()
	writeOneList(doc, data_label[int(prev_id)], fn)

def writeHash(docs_hash, fn, qid=False):
	fd = open(data_path + fn, 'w')
	query = []
	if qid:
		query = ["qid:0"]
	for docid in docs_hash:
		# Ingore the label from the keys and add it manually
		tmp_lst = [str(docs_hash[docid]['label'])] + query + map(lambda k:str(k)+":"+str(docs_hash[docid][k]), sorted(docs_hash[docid].keys()))[:-1]
		fd.write(" ".join(tmp_lst) + "\n")
	fd.close()

def main():
	parser = OptionParser()
	parser.add_option("-f", "--file", dest="ip_fn",
	                  help="select the FILE [train/test]", metavar="FILE")
	parser.add_option("-s", "--stop", dest="stopped",
	                  action="store_true", default=False, help="Use stop words.")
	parser.add_option("-u", "--full", dest="full",
	                  action="store_true", default=False, help="Generate a full dataset.")
	parser.add_option("-n", "--noheader",
					  action="store_true", default=False, help="Do not place header in output.")
	parser.add_option("-j", "--jump", dest="jump",
	                  default=1, help="Jump given number of rows.")
	parser.add_option("-c", "--coo", dest="coo",
					  action="store_true", default=False, help="Save in scipy spare and pickle")
	parser.add_option("-a", "--arff", dest="arff",
	                  action="store_true", default=False, help="Create an arff file.")
	parser.add_option("-v", "--valid", dest="valid",
					  action="store_true", default=False, help="Create a validation set. Only works when input file is train and coo mode on.")
	parser.add_option("-g", "--gforest", dest="gmode",
					  action="store_true", default=False, help="Split the train into a 10%  validation set. Add a query id to the sparse matrix form. (For gforest).")
	(options, args) = parser.parse_args()
	ip_fn = options.ip_fn
	stopped = options.stopped
	full = options.full
	jump = int(options.jump)
	arff = options.arff
	noheader = options.noheader
	coo = options.coo
	gmode = options.gmode
	valid = options.valid
	if jump < 1:
		raise "Invalid jump value provided"
	if ip_fn not in [ training_set, testing_set ]:
		raise "File must be test or train only"

	stop_list = []
	if stopped:
		stop_list = readFile(stop_ids)

	data_set = readCol(ip_fn, stop_list)
	if not stopped:
		assert len(data_set) == len(open(data_path + ip_fn).readlines()), "We shoudnt loose lines while parsing"
	data_label_hash = indexedHasher(ip_fn+".label")

	if full:
		features_num = len(readFile(features))
		if arff:
			header = open(data_path + "header.arff").read()
			file_name = data_path + ip_fn + ".full.arff"
			fd = open(data_path + ip_fn + ".full.arff", 'w')
		else:
			header = ", ".join([ 'col'+str(i) for i in range(1, features_num + 1) ]) + ", label"
			file_name = data_path + ip_fn + ".full.csv"

		fd = open(file_name, 'w')
		if not noheader:
			fd.write(header + "\n")
		fd.close()
		docs = calculateFullList(data_set, data_label_hash, features_num, file_name, jump)
			#writeList(docs, ip_fn + ".full")
	elif coo:
		features_num = len(readFile(features))
		train, validation = calculateSparseDictCOO(data_set, data_label_hash, jump, valid)
		if valid and ip_fn == "train":
			row, col, data, labels = validation
			coo = coo_matrix((data,(row,col)), shape=(max(row)+1, features_num+1))
			cPickle.dump((coo, np.array(labels)), open(data_path  + "validation.sparse.pkl", "w"))

		row, col, data, labels = train
		coo = coo_matrix((data,(row,col)), shape=(max(row)+1, features_num+1))
		cPickle.dump((coo, np.array(labels)), open(data_path + ip_fn + ".sparse.pkl", "w"))
	else:
		docs = calculateSparseDict(data_set, data_label_hash, jump)
		if gmode and ip_fn == "train":
			valid_set = filter(lambda k: k % validation_perc == 0, docs.keys())
			train_set = set(docs.keys()) - set(valid_set)
			valid_data = dict((k,v) for k,v in docs.items() if k in valid_set)
			train_data = dict((k,v) for k,v in docs.items() if k in train_set)
			writeHash(valid_data, "validation.gforest.sparse.dat", qid=gmode)
			writeHash(train_data, "train.gforest.sparse.dat", qid=gmode)
		else:
			name = ip_fn + ".gforest.sparse.dat" if gmode else ".sparse.csv"
			writeHash(docs, name, qid=gmode)



if __name__ == "__main__":
	main()
