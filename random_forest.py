#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


# %%
df = sns.load_dataset("titanic")
df.head()

# %%

features = ["pclass", "sex", "fare", "embarked", "age"]
target = "survived"



# %%
df[features].isnull().sum()
# %%
df[target].isnull().sum()
# %%
x = df[features]
y = df[target]

# %%


num_features = x.select_dtypes(include=["int64","float64"]).columns
cat_features = x.select_dtypes(include=["category","object"]).columns

#%%

num_imputer = SimpleImputer(strategy="mean")
cat_imputer = SimpleImputer(strategy="most_frequent")
x[num_features] = num_imputer.fit_transform(x[num_features])
x[cat_features] = cat_imputer.fit_transform(x[cat_features])


# %%
preproccesor = ColumnTransformer(
    transformers=[
        ("num",StandardScaler(),num_features),
        ("cat",OneHotEncoder(handle_unknown="ignore"),cat_features)
    ]
)
# %%


x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size=0.3,random_state=42
)
# %%

dt = DecisionTreeClassifier(
    max_depth=6,
  
        min_samples_leaf=30,
        class_weight="balanced",
        random_state=42
)
# %%
pipe = Pipeline(
    steps=[
        ("preprocessor",preproccesor),
        ("model",dt)
    ]
)

# %%
pipe.fit(x_train,y_train)

# %%

y_pred_train =  pipe.predict(x_train)
y_pred_test = pipe.predict(x_test)

# %%
print("training accuracy",accuracy_score(y_train,y_pred_train)*100)
print("testing accuracy",accuracy_score(y_test,y_pred_test)*100)


# %%
# random forest
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(
    n_estimators=201,
    bootstrap=True,
    oob_score=True
    
)


# %%
rf_pipepiline  = Pipeline(
    steps=[
        ("preprocessor",preproccesor),
        ("model",rf)
    ]
)

# %%

rf_pipepiline.fit(x_train,y_train)
# %%

y_pred = rf_pipepiline.predict(x_test)


# %%
print("accuracy score",accuracy_score(y_test,y_pred))
print("OOB Score:", rf_pipepiline.named_steps["model"].oob_score_)

# %%
# bagging classifiier
from sklearn.ensemble import BaggingClassifier

base_model = DecisionTreeClassifier(max_depth=4)
bagging = BaggingClassifier(
    base_model,n_estimators=201,
)
bagging.fit(x_train,y_train)


# %%

