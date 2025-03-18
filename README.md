# Final-Capstone-Project-PCMLAI-
This project aims to build and compare classification models to predict whether a client will default on their credit card payment.

<div align='justify'>

## Description of files in repository
* 'train.csv' –the database containing information on client attributes, along with values of the target variable ('credit_card_default').
* ‘FinalCapstone_DelilMartinez.ipynb' -- Jupyter Notebook containing the code and analysis for the project
* 'OptimalDecisionTree.png' – image of the resulting optimal decision tree (the best classifier)
* 'OptimalDecisionTree_nocreditscore.png' – image of the resulting optimal decision tree for the exploratory extension of the project (which ignores the credit_score feature in the dataset)
* 'simplemodels.png', 'improvedmodels.png', 'ensemblemodels.png' - tables containing results from the three different batches of model comparisons
* 'outlier.png' - reference for handling outlier in the dataset

**Link to Jupyter notebook**: [Final Capstone Project Jupyter Notebook](https://github.com/delilx/Final-Capstone-Project-PCMLAI-/blob/main/FinalCapstone_DelilMartinez.ipynb)


## Problem statement: 
Financial lending institutions such as banks and mortgage lenders have as a critical task the decision-making process of approving any particular loan application. The institution charges fees and interest on the loan, so it seems beneficial to grant as many loans as are requested. However, some of the clients end up defaulting on their loans, in which case the institution not only does not reap that profit from the loan, but it does not recover the original amount lent either. Thus, the objective of the institution is to find the right balance in making the decision to grant the loans. 

## Model outcomes or predictions
The purpose of this project is to develop a set of machine learning models that will take clients’ information and predict whether a client with a specific set of attributes or characteristics will default on their loan. That is, we are working with a classification problem. Since the data set actually contains the target feature of whether each client defaulted on their loan, it is considered a supervised classification problem. Such models vary in the quality of their performance (their degree of correctness/success); a major part of the analysis is precisely a comparison of the results. 

## Data acquisition
The original interest resides in being able to explore the connection between clients’ characteristics spanning from the personal such as age and gender to social and economic such as marital status and income but also the existence of previous loans and potential history of credit default. Thus, it was required that the data source(s) include such a variety of features. After searching a variety of sources, a set used for the so-called AmEx CodeLab 2021 contest was selected as it satisfies the requirement. 

**AmExpert 2021 Code Lab** (https://www.kaggle.com/datasets/pradip11/amexpert-codelab-2021/data)

This dataset is associated with a contest hosted by American Express on the platform HackerEarth, in which contestants were to build the best classification model possible using the training set provided and make predictions for the target variable (credit_card_default), which is missing from the data in the test set. The contest was open from November 23 to December 19 of the year 2021. Announcements on the winners are still visible at https://www.hackerearth.com/challenges/new/competitive/amexpert-code-lab/ (though the data sets are not; those were obtained from the Kaggle link posted above).


## Data preprocessing/preparation
The original dataset consists of 45,528 rows  and 19 columns containing the following features:

#### Examining the Features

The following are the descriptions of the features in the dataset:

```
Column Name Description
1. customer_id: unique identification of customer
2. name: name of customer
3. age: age of customer (Years)
4. gender: gender of customer (M or F)
5. owns_car: whether a customer owns a car (Y or N)
6. owns_house: whether a customer owns a house (Y or N)
7. no_of_children: number of children of a customer
8. net_yearly_income: net yearly income of a customer (USD)
9. no_of_days_employed: no. of days employed
10. occupation_type: occupation type of customer
11. total_family_members: no. of family members of customer
12. migrant_worker: customer is migrant worker (Yes or No)
13. yearly_debt_payments: yearly debt of customer (USD)
14. credit_limit: credit limit of customer (USD)
15. credit_limit_used(%): credit limit used by customer
16. credit_score: credit score of customer
17. prev_defaults: no. of previous defaults
18. default_in_last_6months: whether a customer has defaulted (Yes or No)
19. credit_card_default: whether there will be credit card default (Yes or No)  **this is the target variable**
```
### Data Cleanup

The cleanup of the set involved the following:
* It was detected that about 2000 observations had at least one missing values; these were eliminated, leaving a new total of 43,509 rows in the set.
* The categorical variables ‘migrant’ and ‘default6months’ are already coded with numbers in the set; these were forced into the correct type (“object”) so that the rest of the processing would recognize and treat them as the correct type.
* Presence of outliers: one observation was detected with values for income and credit_limit that not only are extremely far away from the rest of the data, but they actually seem to be some kind of error or typo (the income for this observation is recorded at over $140 million!).
<img src="https://github.com/delilx/Final-Capstone-Project-PCMLAI-/blob/main/outlier.png" alt="outlier">

 > Outliers can negatively affect not only the results of the models, but they can also generate an increase in the complexity of the computing tasks and in the time involved in the training of the models. Thus, this one value was removed from the set, leaving a final total of 43,508 observations to work with.
* Additional engineering: ‘income’ and ‘credit_limit’, as most variables that refer to currency amounts, are skewed right. These were successfully processed to correct the skewness by using a logarithmic transformation. Note: this transformation involves only each individual value of the variable at hand which is entered into the logarithm function. Thus, it can be performed before splitting the dataset into training and testing subsets with no risk of so-called “leakage” (where information from the test set “leaks” into the training data, which is problematic as it may lead to overfitting).
* Correlation analysis: although as a general rule it is preferrable to have as many features as possible, it is advisable to examine the correlation between them, as highly correlated features do not add new information individually, yet they may compromise the quality of the results (or at least increase the complexity of the model without necessarily improving results). Not surprisingly, there is a high degree of correlation between the variables 'children' and 'family', and also between 'logincome' and 'logcredit_limit'. Since highly correlated variables convey similar information, keeping all of them can be counterproductive, as the complexity of numerical algorithms increases with higher dimensionality, which could even lead to errors. The features 'family' and 'logincome' are kept for model training, and the other two features are dropped to simplify the analysis.

* Train/test split: The set is to be split prior to any additional processing is carried out, and prior to training the model. Eighty percent of the observations were randomly selected to make up the training set, leaving the remaining 20% for testing purposes. 
* Traditional preprocessing of all features: all numerical features are piped into a standard scaler processor, which effectively transforms values into what is known as z-scores, or number of standard deviations away from the mean of each variable. This is beneficial because (as suggested by the name) it transforms the values of the different features (which can represent vastly different magnitudes) and puts them on a common (“standard”) scale of values. As for categorical features, these are piped through what is known as OneHotEncoder, which effectively creates dummy variables for each possible value of the variable. These are simple binary variables with values 0 and 1 and allow for the numerical computations to run.

## Modeling
The relevant models trained in this project are:
* Individual supervised classification models:
    * logistic regression,
    * k nearest neighbors,
    * decision tree, and
    * support vector machine 
* Ensemble supervised classification models:
    * Random forest, and
    * AdaBoost with tree stumps



## Model evaluation
Before sharing the results of the model assessment and comparison, it is important to note that the set shows a clear class imbalance in the target variable. This, of course, is natural, as one would not expect half of the clients to default on their loans. This imbalance renders the typical performance metric of accuracy inadequate. Instead, an analysis is required to determine what metric to focus on.

|      | Predicted label 0      | Predicted label 1 |
| ------------- | ------------- |---|
|True label 0 | true negative (TN)|false positive (FP) |
| True label 1 | false negative (FN) |true positive (TP) |

<u>Classifier Errors</u>:

* FP = false positive: the model predicts that a client will default on their credit card payment when they actually won't $\rightarrow$ a loan may be denied to a client who would actually be a good payer (lost business!).

* FN = false negative: the model predicts that a client will not default on their payment when in fact they will $\rightarrow$ lost money due to inability to collect loan repayment.

<u> Metric Candidates</u>:

* **Precision** = $\frac{TP}{TP + FP}$ = proportion of all the 'default' predictions that are actually correct.


* **Recall** = $\frac{TP}{TP + FN}$ = proportion of all the actual 'default' labels that the model was able to classify correctly.

* **f1score**  = harmonic mean of precision and recall (a combined metric that aims to balance the previous two).


In the context of this business problem, it seems preferable to select a model that minimizes false negatives, as these represent the situation where the model predicts (incorrectly) that the client will not default; in this scenario, the institution will not recover the amount of money lent out (and it won’t make the anticipated interest gains, at least not in their totality). Hence, the metric to focus on is recall. Nonetheless, it seems reasonable to also keep track of the models’ f1 score, which represents the harmonic average of recall and precision as a secondary performance metric.
The models were compared in batches: an initial set of simple individual models (logistic regression, k nearest neighbors, decision tree and support vector machine), which were later also fine-tuned by working with additional parameters and asking the model to find optimal values for them; and finally, a smaller set of ensemble models, all in an attempt to improve the results. In addition to calculating the chosen performance metrics, it is also useful to note the computational resources required for each model, as this can also inform decision making in case of equal (or close) recall/f1 values for more than one model.


## Findings of the Analysis / Actionable Items
Each batch of models produced a “winner”, as seen in the following tables.

*Simple Models:*

<img src="https://github.com/delilx/Final-Capstone-Project-PCMLAI-/blob/main/simplemodels.png" alt="results for simple models">\\


*Improved ("hyperparametrized) Models:*

<img src="https://github.com/delilx/Final-Capstone-Project-PCMLAI-/blob/main/improvedmodels.png" alt="results for improved models">\\


*Ensemble Models:*

<img src="https://github.com/delilx/Final-Capstone-Project-PCMLAI-/blob/main/ensemblemodels.png" alt="results for ensemble models">\\


Overall, the highest level of recall achieved by any of them is 76.7785\%, which corresponds to a simple logistic regression model, which also turned out to be the fastest model to train. An honorable mention goes to the decision tree model with a slightly lower recall at 74.8993\% and only slightly slower training time. 

### **Logistic Regression: Overall Maximum Recall (optimal model)**
The logistic regression model that maximizes recall, the chosen performance metric, produces a set of estimated coefficients of the logistic regression, which represent factors that affect the odds that a client with a particular set of values for the features in the model will default on their loan. The full list of such coefficients can be found in the Jupyter notebook. While the interpretation of each of these coefficients may be cumbersome, we can highlight the following insights:
* features whose estimated coefficient is positive are associated with increased odds that the value of 'y' will be equal to 1 (that is, that the client will default on their credit card payment); and
* features whose estimated coefficient is negative are associated with decreased odds that the value of 'y' will be equal to 1 (that is, they are associated with lower odds that the client will default).
In both cases, the technical interpretation requires to specify that all other features be held constant/unchanged. With so many encoded features, this makes for a complex comprehension exercise. However, we can extract a couple of examples to illustrate the results:

<u> Positive coefficients: </u>
* 'pct': Holding every other feature constant, for every additional standard deviation above the mean credit utilization, the odds of the client defaulting on their credit card payment increase by $e^{2.115393} \approx 8.29$ or 829\%.
* 'prev_defaults': Holding every other feature constant, for every additional default the client has had in the past, the odds of them defaulting on their credit card payment increase by $e^{1.28354} \approx 6.27$ or 627\%.
* 'occupation_Cooking staff': (this is a dummy/binary/encoded variable, which affects the interpretation slightly). Clients whose occupation is Cooking staff have $e^{0.718487} \approx 2.05$ of approximately double the odds of defaulting on their credit card payment, compared to clients who do not have this occupation.

<u> Negative coefficients: </u>
* 'logincome': for every additional standard deviation above the average logarithm of incomes in the set that a client has, their odds of defaulting on their credit card payment decrease by $1 - e^{-0.007594}\approx 1-0.9924\approx$ 0.007565 or 0.7565%.
* 'occupation_IT staff': (dummy/binary/encoded) Clients whose occupation is IT staff have $1-e^{-0.101813} \approx 1- 0.9032 \approx$ 9.69% lower odds of defaulting on their credit card payment, compared to clients with other occupation.
* 'credit_score': for every additional standard deviation that a client's credit score is above the average, the odds of them defaulting on their credit card payment is reduced by $1-e^{-4.337939}\approx 1-0.013\approx$ 98.7%!

In this model, all the features in the set can be ranked in terms of importance and of their quantitative effect on the odds of a client defaulting on their credit card payment. Thus, it becomes apparent that the features that increase such odds the most are: the percentage of the credit limit used by the client, the number of previous defaults they have on their record, whether they have a default in the previous 6 months, and occupations in the general categories of Cooking/Waiting/Sales/Security. On the other hand, the features that decrease the odds of a client defaulting on their credit card payment the most are: their credit score, not having defaulted on payments in the last 6 months, feminine gender, owning a car and/or a house, and not being a migrant.


### **Decision Tree: Optimal Among Hyparametrized Models**
The main advantage of the decision tree is its simplicity: the only feature that appears in the resulting decision tree is the credit score. Since this is a quantitative feature, the interpretation should be made in terms of standard deviations away from the average credit score of all clients in the set: any client whose credit score is more than 1.328 standard deviations below the average credit score is predicted to default on their credit card payment. This extremely simple model achieves a recall of 74.8993\%, only marginally lower than the maximum recall of all (achieved by the logistic regression) of 76.7785\%, but with the advantage of only needing to know a client’s credit score. While this may appear to be some sort of waste of all the additional information in the dataset, it actually is evidence that the construction of credit scores is sound, and the resulting number actually manages to encapsulate a very well-tuned assessment of a client’s risk of defaulting on a loan.

<img src="https://github.com/delilx/Final-Capstone-Project-PCMLAI-/blob/main/OptimalDecisionTree" alt="decision tree diagram">


**NB:** It is noteworthy that the ensemble models trained with this data set were not able to improve the metric of interest. While in many cases such models produce superior results, interpretation can be really elusive; this is not a problem we have to deal with in this project.

## Additional Exploration
Should it be of interest, the Jupyter notebook also contains an (admittedly somewhat shallow) exploration of the logistic regression and decision tree models under the hypothesis that the credit score were not available in the set. The main features that stand out in such a case are, not surprisingly, the number of previous defaults, and whether the client has had a default in the last 6 months. The results, however, are inferior in terms of recall to the previously discussed optimal models that do benefit from having credit_score as an available feature in the dataset.

<img src="https://github.com/delilx/Final-Capstone-Project-PCMLAI-/blob/main/OptimalDecisionTree_nocreditscore" alt="decision tree diagram without credit score">


## Limitations and Potential Future Steps

The severe imbalance in the target variable affects classification results. In this project I have taken the approach of adjusting the performance metric, though as a future goal it might be desirable to be able to assess a classifier in terms of its accuracy (or "overall correctness"). Thus, a potential avenue to expand/improve the reach of a project such as this one would be to address the class imbalance issue, for instance, through the use of resampling techniques such as random under- or over-sampling, or SMOTE (synthetic minority over-sampling technique).


</div>
