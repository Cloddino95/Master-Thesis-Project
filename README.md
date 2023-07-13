# NLP adoption through Word Embedding in Measuring Human Risk Perception

## Contact information

- Name: Claudio
- Surname: Proietti Mercuri
- Email: cproiettimercuri@gmail.com


## Introduction

This repository contains the Python scripts used in a comprehensive thesis project, which explores risk perception using the 
Word2Vec model. The project replicates and extends the analysis from Bhatia's 2019 study and conducts novel research exploring 
the efficacy of the Word2Vec model in predicting the perception of risk. The scripts are organized into folders representing the 
different components of the study, including Bhatia's analysis, a novel study, residual tests, and T-statistics analysis.

The scripts utilize a downloaded Word2Vec model which is present in the folder “Word2Vec_downloaded”. 
In addition, all scripts require several datasets for execution, which are stored in the accompanying "Data" folder file. 
The readme file provides a detailed breakdown of the purpose and function of each script, allowing for reproducibility and 
further extension of the work.

## Prerequisites/Dependencies

The present work is run on the PyCharm platform and used as an interpreter Anaconda. The adoption of Anaconda was needed to download 
and implement on MacOS the Gensim library (version 4.3.0). 

To run these scripts, you will need to have the following Python libraries installed. If not already installed, you can add 
them using pip or conda. Please refer to each library's installation guide for more detailed instructions. If the versions 
used are known, please ensure your library versions match those specified.

- gensim: (version 4.3.0) A robust open-source vector space modeling and topic modeling toolkit implemented in Python. It uses 
NumPy, SciPy, and optional Cython for performance. In this project, it's used for the implementation of the Word2Vec model.
- numpy: (version 1.23.5) A library for the Python programming language, adding support for large, multi-dimensional arrays and 
matrices, along with a large collection of high-level mathematical functions to operate on these arrays.
- pandas: (version 1.5.3) An open-source data manipulation and analysis library. We use this to handle our data in DataFrame format.
- sklearn: (version 1.0.2) A Machine Learning library in Python. We use it to apply various machine learning algorithms like 
linear regression (Lasso, Ridge), Support Vector Regression (SVR), K Nearest Neighbors Regressor, and K-fold cross-validation. 
It also includes various metrics like Mean Squared Error (MSE).
- tqdm: (version) A fast, extensible progress bar for Python and CLI that allows you to see the progress of your loops.
- matplotlib and seaborn: (version 3.7.1) and (version 0.12.2) They are used for plotting graphs in Python.

Remember to import the Dataset module that is used throughout the scripts. It appears to be a custom module, so ensure you have 
it in your working directory.
To install the packages, you would use pip or conda commands. For example with pip:
`pip install gensim numpy pandas sklearn tqdm matplotlib seaborn`


## Description of each script


### Folder: “Bhatia’s study”

This folder replicates and extends the analysis performed by Bhatia's (2019) paper. It is composed of one folder “PCA” and 
several scripts representing the conducted analysis, below described.  

#### Subfolder: “PCA” 
In this subfolder is run the code for the PCA on the 300-dimensional vectors representing all the risk sources from 
the three datasets. File: “PCA_riskSources.py”

*All the other scripts within Folder*: “**Bhatia’s study**”
-	“Dataset.py”: in this folder are imported all the datasets related to the three studies. To import all of them is needed 
the CSV files are found in an external folder, attached to this guide. Namely, external folder name: “Data” -> open folder: “new_data” 
(inside the new_data folder there are all the CSV files to be imported).
-	“psychometric_1A.py”, “psychometric_1B.py”, “psychometric_2.py” these three scripts run the psychometric experiments for the
three datasets (1A, 1B, 2) 
-	“risk_rating_1A_customized_K_C_AGG.py” (semantic approach for dataset 1A), “risk_rating_1B_customized_K_C_AGG.py” 
(semantic approach for dataset 1B), “risk_rating_2_customized_K_C_AGG.py” (semantic approach for dataset 2). All these three 
scripts run the semantic approach experiments, with “customized” hyperparameters tailored for each ML technique. While Bhatia 
used more generic hyperparameters for all ML techniques. 
-	“dataset2_Bhatia_HYPERPARAMETERS.py” this script performs dataset 2 with the hyperparameters used by Bhatia (2019). 
-	”combined_risk_rating_1A.py”, “combined_risk_rating_1B.py”, “combined_risk_rating_2.py” these three scripts combines the 
the psychometric and semantic approach using therefore 309 dimensions (300 dim from semantic and 9 from psychometric) to show the 
complementary relationship between the two approaches.
-	“pretest_2_checkWordCount.py” you can skip this script. It simply checks whether all the risk sources' names in the pretest 2 
of Bhatia’s paper (2019) correspond to all the risk sources in dataset 2. 
-	“R2_RMSE_study1a1b2_PSY.py” in this script there are all three studies and the psychometric approach together. By running 
this script you can obtain all the metrics (R2 and RMSE) for all the studies conducted in Bhatia except the combined part which 
is done in the next script.
-	“R2_RMSE_study1a1b2_COMBINED.py” same as the above script, in this one is condensed all three scripts referring to the 
combined approach (semantic + psychometric approaches) to get the two metrics.
-	“table_X_y_statistics1A1B2.py” here it is retrieved the summary statistics for the three datasets (1A, 1B, 2) of the semantic 
approach and the dataset for the psychometric approach.



### Folder: “Novel Study”

This folder represents the novel studies conducted on risk perception through the Word2Vec model utilizing datasets 1A, 1B, 
and 2. It is composed of three main folder: “Study1_AssessingVectorEfficacy”, “study2_closest_Words”, “Study3_Algebra_Operations” 
which represent the three main studies. The first study probes the validity and efficacy of the 300-dimensional vector 
representations. The second study aims to enhance the ability of word embeddings to capture the psychological associations that 
individuals attribute to specific words by employing further 300-dimensional vectors representing the n closest words to the 
risk source. The third study leverages word embeddings and algebraic operations on the semantic space to improve model accuracy 
and capture potential semantic differences among geographical areas.

As explained in the Thesis research all the experiments conducted in this section utilize only the best-performing linear model 
(Ridge) and the best-performing non-linear model (SVR-RBF) with a specific hyperparameter deducted via grid-search strategy. 

Outside the three folders, representing the three studies, there are two scripts:
1)	“psy_comparison_LassoRidgeElasticNet_SVR.py” here it is assessed which models and which hyperparameters are the best for the
psychometric approach. In addition to SVR which was already chosen it is tested Lasso, Ridge, and ElasticNet regressors. 
2)	“psy_RidgeSVR_Metrics_and_Predictions.py” here it is retrieved both the metrics (R2 and RMSE) and the predicted values 
(ratings) for the psychometric approach that will be utilized as a benchmark (alongside Bhatia (2019) results for dataset 2) for the novel studies results. This is by utilizing the new approach (i.e. only 2 models and one hyperparameter.

#### Sub-Folder: “Study1_AssessingVectorEfficacy”

-	“Ratings1B_vectors1A.py” this script tests the result of forecasting ratings assigned to risk sources of dataset 1B with the 
vector representing the risk sources of study 1A. 
-	“Ratings1B_VectorsRandom_words.py” this script tests the result of forecasting ratings assigned to risk sources of dataset 1B
with the vector representing random words. 
-	“predict_words(NOratings)_INCONCLUSIVE.py” here the purpose is to forecast the words themselves instead of their rating. 
As such, the dependent variable is categorical rather than a continuous numeric variable. SVM was chosen however it overfit on 
the training set so each word is accurately forecasted but performs poorly on the testing set. It is needed to adjust the 
hypermeter and maybe try a different model (i.e. RNN). 
-	“Bhatia2_SHAP_prediction.py” this script tries to leverage the SHAP technique to obtain the n most influential dimensions 
(from each 300-dimensional vector) to forecast the risk source. However, by using only a sub-portion of the full 300-dimensional
vector the model performance drops because the embedding space representing the distances between vectors is disrupted. 

#### Sub-Folder: “study2_closest_Words”

-	“risk_rating_2_Bhatia.py” it re-run Bhatia’s study for dataset 2 but with the new specifics (only two models and one 
hyperparameter). 
-	“study2_1Closest.py” it utilizes only the most similar word to each risk source to forecast the risk ratings assigned to 
each risk source. 
-	“study2_5Closest.py” it utilizes the five most similar words to each risk source to forecast the risk ratings assigned to 
each risk source. Each vector is concatenated obtaining 1,500 dimensions. 
-	“study2_10Closest.py” it utilizes the ten most similar words to each risk source to forecast the risk ratings assigned to 
each risk source. Each vector is concatenated obtaining 3,000 dimensions. 
-	“study2_Risk_1Closest.py” it adds to the “n” most similar words-vectors, the original vector representing the risk source.
-	“study2_Risk_5Closest.py” it adds to the “n” most similar words-vectors, the original vector representing the risk source.
-	“study2_Risk_10Closest.py” it adds to the “n” most similar words-vectors, the original vector representing the risk source.

#### Sub-Folder: “Study3_Algebra_Operations”

-	“Europe_Perception_TrainTestEU.py” in this script each risk source is forecasted with the EU dataset generated within the 
code. The training and testing is done with the same EU dataset. However, in the code is present also in the original dataset, 
so it is possible to train the model with the original dataset and test with the EU one by simply adjusting the code for the X 
and y datasets used to fit the model. 
-	“US_perception_TrainOrgin_TestUS.py” in this script each risk source is forecasted with the US dataset generated within the 
code. The training is done with the original dataset and the testing is done with the same US dataset.
-	“US_Perception_TrainTestUS.py” in this script each risk source is forecasted with the US dataset generated within the code. 
The training and testing are done with the same US dataset. However, the code is present also in the original dataset, so it is 
possible to train the model with the original dataset and test with the US one by simply adjusting the code for the X and y 
datasets used to fit the model.
-	“US_Risk_5W_perception_trainTestUS_Risk.py” this script is the same as “US_Perception_TrainTestUS.py” however the US dataset
here is formed by adding 5 words about risk and US. This script was changed multiple times to assess different algebraic 
combinations and model limits as described in the thesis research. 
-	“TAILORED_WORDS.py” this script entails the part of the thesis in which is created a “tailored” dataset, where for each 
risk source it is added one vector giving additional information specific to that risk source. Then it is forecasted the 
risk ratings with this tailored dataset. 

- “**Word cloud**” _folder_: has the jpeg images and the scripts (“WordCloud_EU.py” and “WordCloud_USA.py”) to generate the 
word cloud needed to show the semantic shift that occurred through algebraic operations. 



### Folder: “Residuals Test”

In this folder is performed the analysis on the models residual for both the Bhatia study and the novel study.
-	In the folder “Bhatia’s study” there are three scripts (study_1a.py, study_1b.py, and study_2.py) providing the code to 
retrieve the residual for the three studies.
-	In the folder  “NovelStudy2_closest_Words” there are 6 scripts (1word.py, 5words.py, 10words.py, Risk_1Word.py, 
Risk_5Words.py, and Risk_10Words.py) showing the analysis of the residuals conducted on the 6 datasets created for the novel study 2. 

### Folder: “T-Statistic”

In this folder, there are the one-sample t-test, paired t-test, and Shapiro-Wilk test run for both studies (Bhatia (2019) and 
the Novel Study)

#### Sub-Folder “Bhatia’s study”:

-	“T_test_study1A.py”, “T_test_study1B.py”, and “T_Test_study2.py” these three scripts run the one-sample t-test, paired t-test, 
and Shapiro-Wilk test for Bhatia’s studies 1A, 1B, and 2.
-	“t_test_study1A_COMBINED.py”, “T_test_study1B_COMBINED.py”, “T_Test_study2_COMBINED.py” these three scripts run the 
one-sample t-test, paired t-test, and Shapiro-Wilk test for Bhatia’s studies 1A. 1B, and 2 when the psychometric approach 
is combined with the semantic approach. 
 
#### Sub-Folder “Novel Study”:

##### Sub-SubFolder Study 2: 
this folder refers to the study employing the “n” most close words to the risk source for regression purposes.

-	“T_1closest.py”, “T_5close.py”, “T_10close.py” these three scripts run the one-sample t-test, paired t-test, and 
Shapiro-Wilk test for the case in which only the – closest, 5, or 10 most similar words – are employed to forecast the ratings 
assigned to each risk source. 
-	“T_R1close.py”, “T_R5close.py”, “T_R10close.py” these three scripts run the one-sample t-test, paired t-test, and 
Shapiro-Wilk test for the case in which the original vector representing the risk source is concatenated with the – closest, 
5, or 10 most similar words –  to forecast the ratings assigned to each risk source. 
-	“T_psychometric.py” is the benchmark used to perform the t-test (i.e. the difference between the psychometric approach and each 
of the above cases). 

     
##### Sub-Subfolder Study 3: 
this folder refers to the final study in which are employed algebraic operations to assess the model impact and perform semantic and geographic shifts. 

-	“T_Study2_Bhatia.py”, “EU_trainTestEU.py”, “TrainOrigin_TestUS.py” the **first** script refers to the one-sample t-test, 
paired t-test, and Shapiro-Wilk test for Bhatia study 2 used as a second benchmark together with the psychometric approach 
to gauge the semantic performance. While the **second** script pertains to the scenario in which both training and testing are done 
with the same “altered” dataset (EU or US) and **third** script instead show the one-sample t-test, paired t-test, and 
Shapiro-Wilk test for the scenario in which the training is performed on the original dataset and the testing on the 
“altered” dataset (EU or US). 



