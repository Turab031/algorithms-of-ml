#%%
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
# %%
insurance = pd.read_csv("insurance.csv")
# %%
insurance.head()
# %%

x = insurance.drop(columns=["charges"])
y = insurance["charges"]

x = pd.get_dummies(x,columns=["region"],drop_first=False,dtype=int)

x["sex"] = x["sex"].map({"female":1,"male":0})
x["smoker"] = x["smoker"].map({"yes":1,"no":0})

x.head()
# %%
x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size=0.2,random_state=42
)
model = LinearRegression()
model.fit(x_train,y_train)

# %%

y_pred=model.predict(x_test)


# %%
r2 = r2_score(y_test,y_pred)
print(r2)
# %%
x["age_smoker"]=x["age"]*x["smoker"]
x["bmi_smoker"] = x["bmi"]*x["smoker"]
x.head()
# %%

x_train,x_test,y_train,y_test = train_test_split(
    x,y,random_state=42,test_size=0.2

)

model = LinearRegression()
model.fit(x_train,y_train)
# %%
y_pred = model.predict(x_test)
# %%

r2 = r2_score(y_test,y_pred)

print(r2)
# %%
y_train_pred = model.predict(x_train)
r2_train = r2_score(y_train,y_train_pred)
print(r2_train)
print(r2)

  
# %%
