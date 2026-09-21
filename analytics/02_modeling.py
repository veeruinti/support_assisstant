"""Modeling continues only from the committed offline Titanic CSV."""
from pathlib import Path
import joblib, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score, RocCurveDisplay
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

ROOT=Path(__file__).parent; CHARTS=ROOT/"charts"; CHARTS.mkdir(exist_ok=True)
df=pd.read_csv(ROOT/"titanic.csv")
X=df[["pclass","sex","age","sibsp","parch","fare","embarked"]]; y=df.survived
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
print("Class balance:\n", y.value_counts(normalize=True))
numeric=["pclass","age","sibsp","parch","fare"]; categorical=["sex","embarked"]
pre=ColumnTransformer([("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),numeric),("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),categorical)])
models={"Logistic Regression":LogisticRegression(max_iter=1000),"Decision Tree":DecisionTreeClassifier(random_state=42),"Random Forest":RandomForestClassifier(random_state=42,n_estimators=200)}
results=[]; fitted={}
for name, estimator in models.items():
    pipe=Pipeline([("preprocessor",pre),("model",estimator)]).fit(X_train,y_train); fitted[name]=pipe
    pred=pipe.predict(X_test); proba=pipe.predict_proba(X_test)[:,1]
    row={"model":name,"accuracy":accuracy_score(y_test,pred),"precision":precision_score(y_test,pred),"recall":recall_score(y_test,pred),"f1":f1_score(y_test,pred),"auc":roc_auc_score(y_test,proba),"confusion_matrix":confusion_matrix(y_test,pred).tolist()}; results.append(row); print(row)
    RocCurveDisplay.from_predictions(y_test, proba, name=name)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1); plt.tight_layout(); plt.savefig(CHARTS / f"roc_{name.lower().replace(' ', '_')}.png"); plt.close()
pd.DataFrame(results).to_csv(ROOT/"classification_metrics.csv",index=False)
# A readable tree with transformed feature names.
features=fitted["Decision Tree"].named_steps["preprocessor"].get_feature_names_out()
plt.figure(figsize=(24,10)); plot_tree(fitted["Decision Tree"].named_steps["model"],feature_names=features,class_names=["not survived","survived"],filled=True,max_depth=3); plt.savefig(CHARTS/"decision_tree.png",bbox_inches="tight"); plt.close()
# Fit preprocessing only to training fold. SMOTE is applied only after train transformation.
variants={"baseline":RandomForestClassifier(n_estimators=200,random_state=42),"balanced":RandomForestClassifier(n_estimators=200,class_weight="balanced",random_state=42)}
imb=[]
for label, est in variants.items():
    p=Pipeline([("preprocessor",pre),("model",est)]).fit(X_train,y_train); q=p.predict(X_test); imb.append({"variant":label,"precision":precision_score(y_test,q),"recall":recall_score(y_test,q),"f1":f1_score(y_test,q)})
train_array=pre.fit_transform(X_train); test_array=pre.transform(X_test); smote=SMOTE(random_state=42); sx,sy=smote.fit_resample(train_array,y_train); sm=RandomForestClassifier(n_estimators=200,random_state=42).fit(sx,sy); q=sm.predict(test_array); imb.append({"variant":"SMOTE training only","precision":precision_score(y_test,q),"recall":recall_score(y_test,q),"f1":f1_score(y_test,q)})
pd.DataFrame(imb).to_csv(ROOT/"imbalance_metrics.csv",index=False)
grid=GridSearchCV(Pipeline([("preprocessor",pre),("model",RandomForestClassifier(oob_score=True,bootstrap=True,random_state=42))]),{"model__n_estimators":[100,200],"model__max_depth":[None,8],"model__max_features":["sqrt",.7]},cv=3,scoring="f1").fit(X_train,y_train)
best=grid.best_estimator_; print("Best RF:",grid.best_params_,"OOB:",best.named_steps["model"].oob_score_)
joblib.dump(best,ROOT/"best_survival_pipeline.joblib"); assert len(joblib.load(ROOT/"best_survival_pipeline.joblib").predict(X_test.head(2)))==2
# Regression: fare from the other available selected features.
rx=df[["pclass","sex","age","sibsp","parch","embarked"]]; ry=df.fare
rxtr,rxt,rytr,ryt=train_test_split(rx,ry,test_size=.2,random_state=42)
# Regression-specific transformer excludes target fare.
rpre=ColumnTransformer([("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),["pclass","age","sibsp","parch"]),("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),categorical)])
rpipe=Pipeline([("preprocessor",rpre),("model",LinearRegression())]).fit(rxtr,rytr); rp=rpipe.predict(rxt); r2=r2_score(ryt,rp); n=len(ryt); p=rpre.fit(rxtr).transform(rxtr).shape[1]; adj=1-(1-r2)*(n-1)/(n-p-1)
reg={"MAE":mean_absolute_error(ryt,rp),"RMSE":mean_squared_error(ryt,rp)**.5,"R2":r2,"Adjusted_R2":adj}; print("Regression:",reg); pd.DataFrame([reg]).to_csv(ROOT/"regression_metrics.csv",index=False)
comparison = pd.DataFrame(results)[["model", "accuracy", "precision", "recall", "f1", "auc"]]
comparison["metric_group"] = "classification"
comparison = pd.concat([comparison, pd.DataFrame([{"model":"Linear Regression (fare)", "metric_group":"regression", "MAE":reg["MAE"], "RMSE":reg["RMSE"], "R2":reg["R2"], "Adjusted_R2":reg["Adjusted_R2"]}])], ignore_index=True)
comparison.to_csv(ROOT/"model_comparison.csv", index=False)
plt.scatter(rp,ryt-rp,alpha=.6); plt.axhline(0,color="red"); plt.xlabel("Predicted fare"); plt.ylabel("Residual"); plt.savefig(CHARTS/"residuals.png"); plt.close()
