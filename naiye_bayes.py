#%%
import pandas as pd
from sklearn.metrics import precision_score,accuracy_score,recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
# %%
heart_df = pd.read_csv("heart.csv")
# %%
x = heart_df.drop("target",axis=1)
y = heart_df["target"]
# %%
x_train,x_test,y_train,y_test = train_test_split(
    x,y,random_state=42,test_size=0.2
)
# %%
gnb_model = GaussianNB()
gnb_model.fit(x_train,y_train)
y_pred = gnb_model.predict(x_test)
# %%

print("recall score=",recall_score(y_test,y_pred))
print("precision score=",precision_score(y_test,y_pred))
print("accuracy score=",accuracy_score(y_test,y_pred))


# %%
