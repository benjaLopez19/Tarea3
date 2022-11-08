from sklearn.svm import SVC
import sv
from sklearn.model_selection import train_test_split
import numpy as np
import myutility

X,y = sv.select_variables()
X_train, X_test, y_train, y_test = train_test_split(np.transpose(X), y, test_size=0.33, random_state=42)

svc = SVC()

svc.fit(X_train,y_train)

print(svc.score(X_test,y_test))

X,y = myutility.load_data('fuentes\KDDTrain.txt',0)
X_train, X_test, y_train, y_test = train_test_split(np.transpose(X), y, test_size=0.33, random_state=42)

svc.fit(X_train,y_train)
print(svc.score(X_test,y_test))