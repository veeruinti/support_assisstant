# Analytics Pipeline

Run `python 01_eda.py` once with network access, then commit `titanic.csv`; `python 02_modeling.py` reads only that offline fallback. The EDA script prints exact missing percentages and outputs charts/CSVs; retain its console output in your final notebook/report if required.

Missingness decisions follow the rubric: `embarked` and its corresponding `embark_town` values are each 0.22% missing, so those two rows are dropped under the below-5% rule; `age` is 19.87% missing and receives median imputation under the 5--30% rule; `deck` is 77.22% missing and is dropped as unreliable. Fare is right-skewed: its mean (32.10) is above its median (14.45), which is above its mode (8.05). The IQR rule finds 65 age outliers and 114 fare outliers.

`survival_by_sex.png` shows that women had a 74.0% survival rate, versus 18.9% for men. This large gap suggests that sex was strongly associated with survival in this historical dataset.

`survival_by_class.png` shows survival falling from 62.6% in first class to 24.2% in third class. Passenger class likely represents differences in location, access to lifeboats, and socioeconomic advantage.

`fare_survival.png` compares fare distributions by survival status. Higher fares are generally associated with survival, but the strong right tail means fare alone is not enough to predict an outcome.

`age_sex_survival.png` combines age, fare, sex, and outcome. It shows that the sex-based survival pattern remains visible across a broad range of ages and fares, while higher fares cluster more heavily among survivors.

The heatmap uses exactly survived, pclass, age, sibsp, parch, fare. The two strongest absolute off-diagonal correlations are fare/pclass (0.548, negative before taking the absolute value), reflecting more expensive first-class travel, and sibsp/parch (0.415), reflecting family groups travelling together.

The train/test split is stratified because only 38.4% of passengers survived. Pipelines fit imputation, encoding and scaling only on training data. Compare `imbalance_metrics.csv`: prefer the variant with the best recall/F1 trade-off for the survival objective. `model_comparison.csv` deliberately keeps classification metrics and regression metrics in separate columns/groups because they are not on a common scale. Inspect `charts/residuals.png`: a widening/funnel-shaped residual spread indicates heteroscedasticity; an even random band does not.

On this held-out split, the Decision Tree has the best F1 score (0.748) and accuracy (0.816), while Logistic Regression has the best AUC (0.844). I would deploy Logistic Regression when ranking/calibrated probability is important, because its AUC is higher and it is simpler to explain. If the operating point prioritizes the best default F1 score, the Decision Tree is a defensible alternative. The final deployment artifact is the tuned Random Forest pipeline, saved with its preprocessing transformer in `best_survival_pipeline.joblib` and confirmed reloadable on raw input.
