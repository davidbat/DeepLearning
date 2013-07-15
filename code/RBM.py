class RBM():
	data_point = []
	W = []
	k = 0.5
	# initializa with inputs? sparse/full
	def __init__(self, num_features, num_RBMs, nodes_per_lvl = 100):
		self.num_features = num_features
		self.nodes_per_lvl = nodes_per_lvl
		self.num_RBMs = num_RBMs
		self.initialize_W(num_features, num_RBMs)

	def pseudo_code(self, data_point, itr):
		# do some magic to calculate W

		# update W[itr]
		return


	def outer_iterator(self, data_points):
		for dp in data_points:
			for itr in range(num_RBMs):
				self.pseudo_code(data_point, itr)
		return

	def logistic_fnc(self, output_val):
		return (1.0 / (1 + exp(self.k * output_val)))

	def initialize_W(self, num_features, num_RBMs):
		W = self.W
		for i in range(num_RBMs):
			w.append([])
			if i == 0:
				w[i] = [0 for j in range(num_features)]
			else:
				W[i] = [0 for j in range(self.nodes_per_lvl)]


	def apply_W(self, w, p):
		# here i in for a single node in input to those ahead
		# here j is output nodes
		output = [ 0 for i in range(len(w[0])) ]
		for i in range(len(w)):
			for j in range(len(w[i])):
				output[j] += p[i] * w[i][j]

		return map(lambda r: logistic_fnc(r), output)

	def generate_output(self, test_point):
		prev_data = apply_W(W[0], test_point)
		for i in range(1, self.num_RBMs):
			prev_data = apply_W(W[i], prev_data)
			# apply data pt * w

		# use some ecoc map to map the final prev_data to a label
