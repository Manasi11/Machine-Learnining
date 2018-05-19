# Machine-Learnining
Modeling interviewee behavior using Logistic Regression and Decision Tree

Read me file for ML project.

1. Content of submission ZIP file

1.	The zipped file contains an ipynb file: Modelling_Interviewee_behaviours-FINAL.ipynb
The file contains the code for cleaning as well as for the implementation of all the models used (Logistic Regression, Random Forest and Decision Tree).

2.	The zipped file contains 2 CSV(s):
●	Interview.csv (The raw data file)
●	CleanedInterviews-FINAL.csv (The cleaned data file)

3.	Project Report: ML_Project_Report.docx

4.	Better view of Decision Tree: dtreeFINAL.docx

5.	Read me File: Readme File.docx

2. Libraries and modules required for implementing the code:
 
1) Panda (abbreviated as pd),
2) Scikit learn,
3) MatPlotLib (abbreviated as plt),
4) Numpy (abbreviated as np),
5) sklearn - datasets, linear_model,
6) sklearn.metrics - accuracy_score,
7) sklearn.preprocessing - LabelEncoder,
8) statsmodels.api (abbreviated as sm),
9) sklearn.liner_model - LogisticRegression,
10) sklearn.model_selection - train_test_split, cross_val_score
11) sklearn.ensemble - RandomForestClassfier,
12) sklearn.tree - DecisionTreeClassfier, export_graphviz,
13) os and subprocess
14) graphviz
15) sklearn - tree.
 
3. Actual code for implementing above mentioned libraries and modules
 
1) import panda as pd
2) import matlotlib.pyplot as plt
3) import numpy as np
4) from sklearn import datasets, linear_model
5) from sklearn.metrics import accuracy_score
6) from sklearn.preprocessing import LabelEncoder
7) import statsmodels.api (abbreviated as sm)
8) from sklearn.liner_model import LogisticRegression
9) from sklearn.model_selection import train_test_split, cross_val_score
10) from sklearn.ensemble import RandomForestClassfier
11) from sklearn.tree import DecisionTreeClassfier, export_graphviz
12) import os
13) import subprocess
14) import graphviz
15) from sklearn import tree.
 
4. Instructions for implementing the code:
 
6.	The zipped file contains an ipynb file: Modelling_Interviewee_behaviours-FINAL.ipynb
The file contains the code for cleaning as well as for the implementation of all the models used (Logistic Regression, Random Forest and Decision Tree).

1.	The zipped file contains 2 CSV(s):
●	Interview.csv (The raw data file)
●	CleanedInterviews-FINAL.csv (The cleaned data file): The code is/will be exporting this csv file once the data cleaning is done. We have attached this file in submission just for reference.

To run the complete code with cleaning and implementation of models together.

Step 1:
Put the code file “Modelling_Interview_behavior-FINAL.ipynb” and data file “interview.csv” files into a same folder.

Step 2:
Execute the code in Jupyter notebook by executing the Modelling_Interview_behavior - FINAL.ipynb file. 

Step 3:
The decision tree will be exported to another .docx file via the following code snippet: 

dotfile = open("D:/FAKEPATH_REPLACED_BY_ACUTAL_FILEPATH\dtreeFINAL.dot", 'w')

It will generate a dtreeFINAL.docx file. Use the following link to convert it to a better readable image by copying the content of the .docx file onto the textArea upon opening the link. http://www.webgraphviz.com/

5. Reference: The source of the dataset 

Below is the original link from where the data set was taken. We recommend the use of data set attached with submission instead of using it from the below link. The reason for doing this is that, the authors change the data from set time to time. 
https://www.kaggle.com/vishnusraghavan/the-interview-attendance-problem
