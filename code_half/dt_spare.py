import sys
from optparse import OptionParser


def ReadFile_hash(fn, jumps):
	print "Reading file - ", fn
	fd = open(fn)
	lines = fd.readlines()
	hashes = []
	i = -1
	for line in lines:
		i += 1
		if i % jumps != 0:
			continue

		tmp_hash = {}
		line = line.split()
		for item in line [1:]:
			item = item.split(":")
			# keys are ints and values are floats
			tmp_hash[int(item[0])-1] = int(item[1])
		hashes.append(tmp_hash)
	print "Finished reading the data"
	return hashes


def calculate_splits(hashes, index):
	for item in hashes:
		fil


def main():
	parser = OptionParser()
	parser.add_option("-ts", "--test", dest="test_fn",
	                  help="specify the test_file", metavar="FILE")
	parser.add_option("-tr", "--train", dest="train_fn",
	                  help="specify the train file", metavar="FILE")
	parser.add_option("-j", "--jumps", dest="jumps",
	                  help="specify the number of points to skip total/jumps datapoints to consider", metavar="FILE")
	(options, args) = parser.parse_args()



if __name__ == "__main__":
	main()