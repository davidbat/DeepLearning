# Decision tree to handle spare data sets

#!/usr/bin/python
import math
import operator
def readfile(filename):
  fd = open(filename)
  features = []
  label = []
  for line in fd.readlines():
    line = line.strip().split()
    if line[len(line) - 1] not in label:
      label.append(line[len(line) - 1])
    features.append(map(lambda x:float(x),line))
  return features

class LeafNode:
  left , right, data = None, None, 0.0

  def __init__(self, data):
    # initializes the data members
    self.left = None
    self.right = None
    self.data = data

class PredicateNode:
  left , right, feature_id, condition = None, None, 0.0, 0.0

  def __init__(self, ids, val):
    # initializes the data members
    self.left = None
    self.right = None
    self.feature_id = ids
    self.condition = val

class Tree:
  def __init__(self):
    # initializes the root member
    self.root = None

  def addLeafNode(self, data):
    # creates a new node and returns it
    return LeafNode(data)

  def addPredNode(self, feature, value):
    # creates a new node and returns it
    return PredicateNode(feature, value)

  def insert(self, root, data):
    # inserts a new data
    if root == None:
      # it there isn't any data
      # adds it and returns
      return self.addNode(data)
    else:
      # enters into the tree
      if data <= root.data:
        # if the data is less than the stored one
        # goes into the left-sub-tree
        root.left = self.insert(root.left, data)
      else:
        # processes the right-sub-tree
        root.right = self.insert(root.right, data)
      return root

  def uniq_labels(self, Y):
    uniq_Y = {}
    for label in Y:
      if label in uniq_Y.keys():
        uniq_Y[label] += 1
      else:
        uniq_Y[label] = 1
    return uniq_Y

  def calculate_ig_ent(self, X, split_row):
    # p * log(1 / p)
    # for x-merging we will need to handle a list in labels spot
    entropy_left = 0
    entropy_right = 0
    parent_entropy = 0
    items_num = len(X)
    right_items_num = items_num - split_row
    Y = self.uniq_labels(map(lambda row:row[-1] ,X))
    for label in Y.keys():
      parent_entropy += (float(Y[label]) / len(X) * math.log(len(X) / float(Y[label]), 2))
      #parent_entropy += (float(Y[label]) / len(X)) * (1.0 - float(Y[label]) / len(X))
    #print parent_entropy, Y
    Y = self.uniq_labels(map(lambda row:row[len(row)-1] ,X[0:split_row]))
    for label in Y.keys():
      entropy_left += float(Y[label]) / split_row * math.log(split_row / float(Y[label]), 2)
      #entropy_left +=  (float(Y[label]) / split_row) * (1.0 - float(Y[label]) / split_row )
    #print entropy_left, Y
    Y = self.uniq_labels(map(lambda row:row[len(row)-1] ,X[split_row:]))
    for label in Y.keys():
      entropy_right += float(Y[label]) / right_items_num * math.log(right_items_num / float(Y[label]),2)
      #entropy_right += (float(Y[label]) / right_items_num) * (1.0 - float(Y[label]) / right_items_num)
    #print entropy_right, Y
    IG = float(split_row) / items_num * (parent_entropy - entropy_left) + right_items_num /items_num * (parent_entropy - entropy_right)
    return IG

  # def calculate_ig_gini(self, X, split_row):
  #   # 1 - sum(p^2)
  #   # for x-merging we will need to handle a list in labels spot
  #   entropy_left = 1
  #   entropy_right = 1
  #   parent_entropy = 1
  #   items_num = len(X)
  #   right_items_num = items_num - split_row
  #   Y = self.uniq_labels(map(lambda row:row[-1] ,X))
  #   for label in Y.keys():
  #     parent_entropy -= (float(Y[label]) / len(X)) ** 2
  #     #parent_entropy += (float(Y[label]) / len(X)) * (1.0 - float(Y[label]) / len(X))
  #   #print parent_entropy, Y
  #   Y = self.uniq_labels(map(lambda row:row[len(row)-1] ,X[0:split_row]))
  #   for label in Y.keys():
  #     entropy_left -= (float(Y[label]) / split_row) ** 2
  #     #entropy_left +=  (float(Y[label]) / split_row) * (1.0 - float(Y[label]) / split_row )
  #   #print entropy_left, Y
  #   Y = self.uniq_labels(map(lambda row:row[len(row)-1] ,X[split_row:]))
  #   for label in Y.keys():
  #     entropy_right -= (float(Y[label]) / right_items_num) ** 2
  #     #entropy_right += (float(Y[label]) / right_items_num) * (1.0 - float(Y[label]) / right_items_num)
  #   #print entropy_right, Y
  #   IG = float(split_row) / items_num * (parent_entropy - entropy_left) + right_items_num /items_num * (parent_entropy - entropy_right)
  #   return IG

  # def calculate_ig_ms(self, X, split_row):
  #   # mse-left = for all left-val: (left-avg - val)^2
  #   # mse-left * p-left + mse-right * p-right
  #   # for x-merging we will need to handle a list in labels spot
  #   entropy_left = 1
  #   entropy_right = 1
  #   parent_entropy = 1
  #   items_num = float(len(X))
  #   right_items_num = items_num - split_row
  #   #if split_row == 0:
  #   #  print split_row#, X
  #   avg_Y_left = sum(map(lambda row:row[-1], X[0:split_row])) / split_row
  #   avg_Y_right = sum(map(lambda row:row[-1], X[split_row:])) / right_items_num
  #   MSE = 0
  #   left_MSE = sum(map(lambda row:(avg_Y_left - row[-1]) ** 2 ,X[0:split_row])) / split_row
  #   right_MSE = sum(map(lambda row:(avg_Y_right - row[-1]) ** 2 ,X[split_row:])) / right_items_num
  #   #return (left_MSE * split_row + right_MSE * right_items_num) / items_num
  #   return left_MSE + right_MSE

  # Merge contiguous X
  # X must be sort on column 'col'
  def Merge(self, X, col):
    X2 = {}
    dictlist = []
    itr = 0
    for row in X:
      if row[col] not in X2:
        X2[row[col]] = itr
      itr += 1
    #for key, value in X2.iteritems():
    #  dictlist.append([key, value])
    #return sorted(dictlist, key=lambda row: row[1])
    return  sorted(X2.iteritems(), key=operator.itemgetter(1))


  def DT_split(self, cutoff, root, path, X, ig = "entropy"):
    #print X
    #stop condition
    if len(X) == 0:
      #print "here"
      return
    if len(X) <= cutoff:
      if path == "left":
        root.left = self.addLeafNode(map(lambda row:row[len(row) - 1], X))
      else:
        root.right = self.addLeafNode(map(lambda row:row[len(row) - 1], X))
      #print "Here here"
      return

    # for every feature
    #last column is labels
    best_IG = -9999
    best_v = -9999
    best_f = -9999
    best_split = -9999
    # last column is a label
    for feature_num in range(0, len(X[0]) - 1):
      #print feature_num
      X = sorted(X, key=lambda row: row[feature_num])
      # Gotta do some sort of merging for same value of X
      #print X
      X2 = self.Merge(X, feature_num)
      #print "This is X2 - \n", X2
      # X2 = [[feature_value, index in X] ...]
      prev_val = -9999.0
      cur_row = 0
      # for every split
      #for value in map(lambda row:row[feature_num], X):
      for value, X_index in map(lambda row:[row[0], row[1]], X2):
        #if X_index == 0:
        #  print X2
        cur_row = X_index
        if prev_val == -9999.0:
          prev_val = value
          #cur_row += 1
          continue
        split_value = (prev_val + value) / 2
        #calculate entropy
        if ig == "entropy":
          IG = self.calculate_ig_ent(X, cur_row)
        elif ig == "gini":
          IG = self.calculate_ig_gini(X, cur_row)
        else:
          # mult by -1 to change minimize to maximize for MSE
          IG = -1.0 * self.calculate_ig_ms(X, cur_row)
        #print "IG = ", IG
        # update best split if better entropy
        if IG > best_IG:
          best_v = split_value
          best_split = cur_row
          best_IG = IG
          best_f = feature_num
        #cur_row += 1
    #save best split in a PredicateNode
    #something wrong below...
    # need to change root and not make new node
    X = sorted(X, key=lambda row: row[best_f])
    if best_IG == 0:
      #dont split
      if path == "left":
        root.left = self.addLeafNode(map(lambda row:row[len(row) - 1], X))
      elif path == "right":
        root.right = self.addLeafNode(map(lambda row:row[len(row) - 1], X))
      else:
        print "Tree cannot have only 1 node"
      return
    #print "BEST STUFFF ", best_IG,best_f,best_v
    if path == "left":
      root.left = self.addPredNode(best_f, best_v)
      root = root.left
    elif path == "right":
      root.right = self.addPredNode(best_f, best_v)
      root = root.right
    else:
      #print "Here"
      root.feature_id = best_f
      root.condition = best_v
    self.DT_split(cutoff, root, "left", X[0:best_split], ig)
    self.DT_split(cutoff, root, "right", X[best_split:], ig)

  def printTree(self, root, level = 0):
    # prints the tree path
    if root == None:
      pass
    else:
      self.printTree(root.left, level + 1)
      if isinstance(root, LeafNode):
        print "leaf at ", level, " - ",root.data
      else:
        print "pred at ", level, " - ",root.feature_id,root.condition
      self.printTree(root.right, level + 1)

  def lookup(self, root, X):
    # looks for a value into the tree
    if root == None:
      return 0
    elif isinstance(root, LeafNode):
      return root.data
    else:
      #print root.feature_id, root.condition, X[root.feature_id]
      if X[root.feature_id] < root.condition:
        # left side
        return self.lookup(root.left, X)
      else:
        # right side
        return self.lookup(root.right, X)

  def MSE(self, root, filename):
    sums = 0.0
    cnt = 0
    for line in open(filename).readlines():
      line = line.split()
      expected = float(line[-1])
      obtained = self.lookup(root, map(lambda x:float(x),line))
      obtained = sum(obtained) / len(obtained)
      MSE = (expected - obtained) ** 2
      sums += MSE
      cnt += 1
    return sums / cnt

def method(calculate = "entropy"):
  print "\n\nCalculating DT based on", calculate
  filename = "training"
  features = readfile(filename)
  tr = {}
  root = {}
  for cutoff in [1,2,3,4,5]:
    tr[cutoff] = Tree()
    root[cutoff] = tr[cutoff].addPredNode(-1,-1)
    print "Using leaf node cutoff as - ", cutoff
    tr[cutoff].DT_split(cutoff, root[cutoff], "start", features, calculate)
    print "Average MSE training - ", tr[cutoff].MSE(root[cutoff], "training")
    print "Average MSE test - ", tr[cutoff].MSE(root[cutoff], "test")


if __name__ == "__main__":
  method("entropy")
  #method("gini")
  #method("MSE")
