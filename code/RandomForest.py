from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import cPickle

X, y = cPickle.load(open("../data/train.sparse.pkl"))
X = X.toarray()

clf = RandomForestClassifier(n_estimators=1000, min_samples_split=3)
scores = cross_val_score(clf, X, y)
print scores.mean()
