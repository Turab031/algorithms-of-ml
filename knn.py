#%%
import pandas as pd
from sklearn.metrics import precision_score,recall_score,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# %%
heart_df = pd.read_csv("heart.csv")
x = heart_df.drop("target",axis=1)
y = heart_df["target"]

# %%
x_train,x_test,y_train,y_test = train_test_split(
    x,y,random_state=42,test_size=0.2
)
# %%
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
# %%
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(x_train_scaled,y_train)
y_pred = knn_classifier.predict(x_test_scaled)

print("recall score=",recall_score(y_test,y_pred))
print("precision score=",precision_score(y_test,y_pred))
print("accuracy score=",accuracy_score(y_test,y_pred))




# %%
# %%
# for k = 5
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(x_train_scaled,y_train)
y_pred = knn_classifier.predict(x_test_scaled)

print("recall score=",recall_score(y_test,y_pred))
print("precision score=",precision_score(y_test,y_pred))
print("accuracy score=",accuracy_score(y_test,y_pred))

# %%
# for k = 7
knn_classifier = KNeighborsClassifier(n_neighbors=7)
knn_classifier.fit(x_train_scaled,y_train)
y_pred = knn_classifier.predict(x_test_scaled)

print("recall score=",recall_score(y_test,y_pred))
print("precision score=",precision_score(y_test,y_pred))
print("accuracy score=",accuracy_score(y_test,y_pred))

# %%
from sklearn.model_selection import GridSearchCV

# %%

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

classifier = KNeighborsClassifier()

param_grid = {
    "n_neighbors": [3,5,7,9]
}

classifier_cv = GridSearchCV(
    estimator=classifier,
    param_grid=param_grid,
    cv=5
)

classifier_cv.fit(x_train_scaled, y_train)

print("Best Parameters:", classifier_cv.best_params_)
print("Best CV Score:", classifier_cv.best_score_)

y_pred = classifier_cv.predict(x_test_scaled)

print("Recall Score =", recall_score(y_test, y_pred))
print("Precision Score =", precision_score(y_test, y_pred))
print("Accuracy Score =", accuracy_score(y_test, y_pred))
# %%
