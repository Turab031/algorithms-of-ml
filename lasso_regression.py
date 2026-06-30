#%% 
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
import numpy as np



import pandas as pd
import seaborn as sns

# %%

insurance = pd.read_csv("insurance.csv")
x = insurance.drop(columns=["charges"])
y = insurance["charges"]

x = pd.get_dummies(x,columns=["region"],drop_first=False,dtype=int)

x["sex"] = x["sex"].map({"female":1,"male":0})
x["smoker"] = x["smoker"].map({"yes":1,"no":0})
x["age_smoker"]=x["age"]*x["smoker"]
x["bmi_smoker"] = x["bmi"]*x["smoker"]





#%%




x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size=0.2,random_state=42


)
# %%
alphas = [0.001,0.1,1,2,5,10,20,30,40,50,1000]
mses=[]

for a in alphas:
    lasso_model = Lasso(alpha = a)
    lasso_model.fit(x_train,y_train)
    y_pred=lasso_model.predict(x_test)
    mse=mean_squared_error(y_test,y_pred)

    print(f"mse for alpha ={a} :",mse)
    mses.append(mse)


sns.lineplot(x=alphas,y=mses,marker="o")
# %%
from sklearn.linear_model import LassoCV
# %%

a = [0.001,0.1,1,2,5,10,20,30,40,50,1000] 

lasso_cv_model = LassoCV(
    alphas =a,
    cv=5,
    max_iter=1000,
    random_state=42

)

lasso_cv_model.fit(x_train,y_train)
print("best alpha model:",lasso_cv_model.alpha_)


y_pred = lasso_cv_model.predict(x_test)
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
print("mse =",mse)
print("r2=",r2)
# %%
