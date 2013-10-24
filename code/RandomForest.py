from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import cPickle

X, y = cPickle.load(open("../data/RF.train.sparse.pkl"))
X = X.toarray()
Xtest, ytest = cPickle.load(open("../data/RF.test.sparse.pkl"))
Xt = Xtest.toarray()

clf = RandomForestClassifier(n_estimators=500, min_samples_split=3)
#print "Doing a cross validation"
#scores = cross_val_score(clf, X, y)
#print scores.mean()

print "Fitting"
clf.fit(X, y)


print "Testing"
print clf.score(Xt, ytest)
