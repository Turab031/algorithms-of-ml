#%%
import pandas as pd
import seaborn as sns
# %%
insurance = pd.read_csv("insurance.csv")

# %%
insurance
# %%
sns.scatterplot(x=insurance["bmi"],y= insurance["charges"],hue=insurance["smoker"])


# %%
x = insurance.drop(columns=["charges","region"])
y=insurance["charges"]

x["sex"] = x["sex"].map({"female":1,"male":0})
x["smoker"]= x["smoker"].map({"yes":1,"no":0})
# %%
x.head()
# %%


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# %%
x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size=2,random_state=42
)


x_train.head()
# %%
model = LinearRegression()
model.fit(x_train,y_train)


# %%
y_pred=model.predict(x_test)

y_pred
# %%
y_test
# %%

from sklearn.metrics import r2_score
r2=r2_score(y_test,y_pred)


# %%
n = x_test.shape[0]
p = x_test.shape[1]

adjusted_r2 = 1-((1-r2)*(n-1)/(n-p-1))
adjusted_r2
# %%
