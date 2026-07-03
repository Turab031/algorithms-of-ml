#%%
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score
# %%
heart = pd.read_csv("heart.csv")
heart
# %%
heart.head()
# %%
x = heart.drop(columns="target")
y = heart["target"]
# %%
x.head()
# %%
x_train,x_test,y_train,y_test = train_test_split(
    x,y,random_state=42,test_size=0.2
)

# %%
model = LogisticRegression(max_iter=5000)
model.fit(x_train,y_train)
# %%
y_pred = model.predict(x_test)


# %%
print("Accuracy=",accuracy_score(y_test,y_pred)*100,"%")
print("precison=",precision_score(y_test,y_pred)*100,"%")
# %%
from sklearn.preprocessing import StandardScaler


# %%
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# %%
model.fit(x_train,y_train)
# %%
y_pred = model.predict(x_test)

# %%

print("Accuracy:",accuracy_score(y_test,y_pred)*100,"%")
print("Precision:",precision_score(y_test,y_pred)*100,"%")
# %%
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
# %%
confusion_matrix(y_test,y_pred)


# %%
print("accuracy score=",accuracy_score(y_test,y_pred))
print("precision score=",precision_score(y_test,y_pred))
print("recall score=",recall_score(y_test,y_pred))
print("f1 score=",f1_score(y_test,y_pred))




# %%
