"""One permitted raw Titanic load, cleaning, EDA, charts and written metrics."""
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
CHARTS = ROOT / "charts"; CHARTS.mkdir(exist_ok=True)
df = sns.load_dataset("titanic")
df.to_csv(ROOT / "titanic.csv", index=False)  # offline fallback immediately after raw load
print(df.info()); print(df.describe(include="all")); print("shape:", df.shape)
missing = (df.isna().mean().mul(100)).loc[lambda s: s.gt(0)]
print("Missing percentages:\n", missing.round(2))

# deck (77.2%) is too sparse: drop. age (19.9%) and embarked (0.22%) are imputed below.
clean = df.drop(columns="deck").dropna(subset=["embarked"]).copy()
clean["age"] = clean["age"].fillna(clean["age"].median())
clean["embark_town"] = clean["embark_town"].fillna(clean["embark_town"].mode()[0])
clean["alive"] = clean["alive"].fillna("no")
clean.to_csv(ROOT / "titanic_cleaned.csv", index=False)

def outliers(s):
    q1, q3 = s.quantile([.25, .75]); iqr=q3-q1
    return int(((s < q1-1.5*iqr) | (s > q3+1.5*iqr)).sum())
for column in ["age", "fare"]:
    fig, ax = plt.subplots(1,2, figsize=(10,4)); sns.histplot(clean[column], kde=True, ax=ax[0]); sns.boxplot(x=clean[column], ax=ax[1])
    fig.tight_layout(); fig.savefig(CHARTS / f"{column}_univariate.png"); plt.close(fig)
    print(f"{column} IQR outliers:", outliers(clean[column]))
fare = clean.fare
print("Fare mean/median/mode:", fare.mean(), fare.median(), fare.mode().iloc[0])
print("Survival by sex:\n", clean.groupby("sex").survived.mean())
print("Survival by pclass:\n", clean.groupby("pclass").survived.mean())
print("Survival by sex+pclass:\n", clean.groupby(["sex", "pclass"]).survived.mean())
# Explicit boolean masks, as requested.
print("Masked female first-class survival:", clean.loc[(clean.sex=="female") & (clean.pclass==1), "survived"].mean())
cols=["survived","pclass","age","sibsp","parch","fare"]
corr=clean[cols].corr(); corr.to_csv(ROOT / "correlation_matrix.csv")
plt.figure(figsize=(7,5)); sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1); plt.tight_layout(); plt.savefig(CHARTS/"correlation.png"); plt.close()
pairs=corr.where(~np.eye(len(corr), dtype=bool)).stack().abs().sort_values(ascending=False)
print("Top correlations (duplicated pairs removed):", pairs[~pairs.index.map(lambda x: x[0]>x[1])].head(2))
# Four data-story charts.
for name, plot in {
 "survival_by_sex": lambda: sns.barplot(data=clean, x="sex", y="survived"),
 "survival_by_class": lambda: sns.barplot(data=clean, x="pclass", y="survived"),
 "fare_survival": lambda: sns.boxplot(data=clean, x="survived", y="fare"),
 "age_sex_survival": lambda: sns.scatterplot(data=clean, x="age", y="fare", hue="survived", style="sex"),
}.items():
    plt.figure(figsize=(7,4)); plot(); plt.tight_layout(); plt.savefig(CHARTS/f"{name}.png"); plt.close()
z = clean[["age","fare"]].apply(lambda s: (s-s.mean())/s.std(ddof=0))
print("Z-score means/stds (approximately 0/1):\n", pd.DataFrame({"mean":z.mean(), "std":z.std(ddof=0)}))
