import RBM

def main():
	parser = OptionParser()
	parser.add_option("-tr", "--train", dest="train",
	                  help="specify the training set file", metavar="FILE")
	parser.add_option("-s", "--sparse", dest="sparse",
	                  default=1, help="Data is a sparse file")
	parser.add_option("-tr", "--test", dest="test",
	                  help="specify the testing set file", metavar="FILE")


#init

# build data struct

# pre train using RBM
