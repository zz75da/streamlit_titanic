import joblib
joblib.dump(clf, "model")
import pickle
pickle.dump(clf, open("model", 'wb'))
#Import du classifieur clf précédemment entrainé et précédemment enregistré sous le nom "model" avec Joblib :
joblib.load("model")
#Import du classifieur clf précédemment entrainé et précédemment enregistré sous le nom "model" avec Pickle :
loaded_model = pickle.load(open("model", 'rb'))