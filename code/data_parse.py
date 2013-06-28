import sys
from optparse import OptionParser

data_path = "../data/"

training_set = "train"

testing_set = "test"

stop_ids = "stop_word_ids.txt"
features = "vocabulary.txt"


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
		if data_set[i][0] not in docs:
			if not data_set[i][0] % jump == 0:
				continue
			docs[data_set[i][0]] = { int(data_set[i][1]): data_set[i][2] , 'label': data_label[int(data_set[i][0])]}
		else:
			docs[data_set[i][0]][int(data_set[i][1])] = data_set[i][2]
	return docs

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
			doc[int(data_set[i][1])] = data_set[i][2]
			# import pdb; pdb.set_trace()
			# doc[features_num] = data_label[int(data_set[i][0])]
			# tmp_lst = [ '0' for i in range(features_num + 1) ]
			# docs[data_set[i][0]] = tmp_lst
			# docs[data_set[i][0]][int(data_set[i][1])] = data_set[i][2]
			# docs[data_set[i][0]][features_num] = data_label[int(data_set[i][0])]
		else:
			try:
				doc[int(data_set[i][1])] = data_set[i][2]
			except:
				import pdb; pdb.set_trace()
	writeOneList(doc, data_label[int(prev_id)], fn)

def writeHash(docs_hash, fn):
	fd = open(data_path + fn, 'w')
	for docid in docs_hash:
		tmp_lst = map(lambda k:str(k)+":"+str(docs_hash[docid][k]), sorted(docs_hash[docid].keys()))[:-1] + [str(docs_hash[docid]['label'])]
		fd.write(", ".join(tmp_lst) + "\n")
	fd.close()

def main():
	parser = OptionParser()
	parser.add_option("-f", "--file", dest="ip_fn",
	                  help="select the FILE [train/test]", metavar="FILE")
	parser.add_option("-s", "--stop", dest="stopped",
	                  action="store_true", default=False, help="Use stop words")
	parser.add_option("-u", "--full", dest="full",
	                  action="store_true", default=False, help="Generate a full dataset")
	parser.add_option("-j", "--jump", dest="jump",
	                  default=1, help="Jump given number of rows")
	parser.add_option("-a", "--arff", dest="arff",
	                  action="store_true", default=False, help="Create an arff file")
	(options, args) = parser.parse_args()
	ip_fn = options.ip_fn
	stopped = options.stopped
	full = options.full
	jump = int(options.jump)
	arff = options.arff
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
		fd.write(header + "\n")
		fd.close()
		docs = calculateFullList(data_set, data_label_hash, features_num, file_name, jump)
			#writeList(docs, ip_fn + ".full")
	else:
		docs = calculateSparseDict(data_set, data_label_hash, jump)
		writeHash(docs, ip_fn + ".sparse.csv")



if __name__ == "__main__":
	main()
