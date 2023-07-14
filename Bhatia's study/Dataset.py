import pandas as pd
# import numpy as np

"""all the below dataset have been previously cleaned in R the code and the file are in the *data* folder -> *new_data* """

# --------------------------------------------
psychometric_1A = pd.read_csv('/Users/ClaudioProiettiMercuri_1/Desktop/MS_Business intelligence/thesis/Datenmodell '
                              'Bhatia/Data/new_data/Psychometric_1A.csv')
psychometric_1B = pd.read_csv('/Users/ClaudioProiettiMercuri_1/Desktop/MS_Business intelligence/thesis/Datenmodell '
                              'Bhatia/Data/new_data/Psychometric_1B.csv')

risk_ratings_1A = pd.read_csv('/Users/ClaudioProiettiMercuri_1/Desktop/MS_Business intelligence/thesis/Datenmodell '
                              'Bhatia/Data/new_data/Risk_Ratings_1A.csv')
risk_ratings_1A.replace(to_replace={'insecticide  - [Field-1]': 'insecticide'}, inplace=True)

risk_ratings_1B = pd.read_csv('/Users/ClaudioProiettiMercuri_1/Desktop/MS_Business intelligence/thesis/Datenmodell '
                              'Bhatia/Data/new_data/Risk_Ratings_1B.csv')

# --------------------------------------------
risk_ratings_2 = pd.read_csv('/Users/ClaudioProiettiMercuri_1/Desktop/MS_Business intelligence/thesis/Datenmodell '
                             'Bhatia/Data/new_data/Risk_Ratings_2.csv')

psychometric_2 = pd.read_csv('/Users/ClaudioProiettiMercuri_1/Desktop/MS_Business intelligence/thesis/Datenmodell '
                             'Bhatia/Data/new_data/Psychometric_2.csv')

pretest_2 = pd.read_csv('/Users/ClaudioProiettiMercuri_1/Desktop/MS_Business intelligence/thesis/Datenmodell '
                        'Bhatia/Data/new_data/Pretest_2.csv')

# --------------------------------------------
study_3 = pd.read_csv('/Users/ClaudioProiettiMercuri_1/Desktop/MS_Business intelligence/thesis/Datenmodell '
                      'Bhatia/Data/new_data/Study_3.csv')

# -------------------------------------------- drop the first column

psychometric_1A = psychometric_1A.drop(psychometric_1A.columns[0], axis=1)
psychometric_1B = psychometric_1B.drop(psychometric_1B.columns[0], axis=1)
risk_ratings_1A = risk_ratings_1A.drop(risk_ratings_1A.columns[0], axis=1)
risk_ratings_1B = risk_ratings_1B.drop(risk_ratings_1B.columns[0], axis=1)
risk_ratings_2 = risk_ratings_2.drop(risk_ratings_2.columns[0], axis=1)
psychometric_2 = psychometric_2.drop(psychometric_2.columns[0], axis=1)
pretest_2 = pretest_2.drop(pretest_2.columns[0], axis=1)
study_3 = study_3.drop(study_3.columns[0], axis=1)


"""
•	File name: “Study 1A and 1B - Risk Ratings” ✅

73 participants study 1A 
79 participants study 1B
Each participant rated each risk source.
125 risk sources 
Valued from -100 (safe) --> +100 (risky)
TODO: in the paper it predicts yi (ratings) both aggregate lvl (mean ✅) and individual lvl. check if indivdual means i have to do a loop of the model in which the dependendt variable yi is each time a single rating and the model kind of calibrate on the base of each iteration or all the ratings for that source

STIMULI:
1A: It contained various common technologies, emerging technologies, military technologies, household appliances, energy sources, drugs, and medical procedures.
1B: containS various hobbies, sports, and occupations.
---------------------------------------------------------------------------------------- 
•	File name: “Study 1A and 1B – Psychometric Dimensions_Dimensional ratings task” ✅

In study 1A 
75 participants
125 risk sources for this task. 

Each participant rated (1 to 7) 15 randomly selected risk sources on all nine risk dimensions generating an average of:
9 ratings per dimension per risk source. 
125(risk sources)*9(dimensions) = 1125 columns
150 rows (75 x 2)

The same settings for study 1B.
----------------------------------------------------------------------------------------
•	File name: “Study 2 - Risk Ratings” ✅

Primary study!!!
300 participants & 200 different risk sources (Tot risk sources: 200)
Each participant was given 100 randomly risk sources

Each risk source has an average of 150 ratings Valued from -100 (safe) to +100 (risky)
200 columns (risk sources)
300 rows (participants)

STIMULI:
consisted of a set of 200 risk sources, of varying risk levels. This set was generated in the study 2 pretest.
The items obtained in this pretest were pruned to select the 200 most frequently listed risk sources (also present in the Word2Vec vocabulary).
so from the 780 stimuli in the pretest we selected the 200 most frequent stimuli

----------------------------------------------------------------------------------------
•	File name: “Study 2 - Psychometric Dimensions (dimensional ratings task)” ✅

Primary study
there were 301 participants 
200 risk sources 
Each participant rated 20 randomly selected risk sources on all nine risk dimensions generating an average of 
30 ratings per risk source.
200*9 = 1800 columns 
300 rows (participants) 
----------------------------------------------------------------------------------------
•	File name: “Study 2 - Pretest” ✅

Prior to running study 2, we ran a study 2 pretest on 52 participants. 
The participants in this pretest were each asked to generate 15 everyday sources of risk: 
(5 with high risk, 5 with medium risk, and 5 with low risk). 
We used the participant-generated risk sources in this pretest as our stimuli in study 2

Tot stimuli 780 (data) = 52(participant-rows) x 15(5 high – 5med-5 low)
15 columns
52 rows (participants)

I WILL NOT USE THIS!!! it exists as a source (stimuli) of risk sources for study 2, in which, out of 780 risk sources generated in this pretest, were chosen the 200 most popular and contained in word2vec vocabulary
----------------------------------------------------------------------------------------
•	File name: “Study 3” ✅

49 participants 
were asked to list 3 words that they associated with the nine dimensions used in risk dimension ratings task. 
54 columns = 9 (dimensions) x 2(“opposite” [e.g.: voluntary / no voluntary]) x 3 (words)
2646 data = (9x2)_18 x (participant)_49 x (words)_3

each participant list three words that first came to their mind when thinking of a risk source with that description. 
For each dimension we used two descriptions, one corresponding to high values on that dimension and the other 
corresponding to low values on that dimension. 
Thus, for example, for the voluntariness dimension we used “a risk source that individuals are exposed to voluntarily” 
and “a risk source that individuals are exposed to involuntarily” as the two descriptions, and each of these two 
descriptions served as a separate cue in the free association task.

"""