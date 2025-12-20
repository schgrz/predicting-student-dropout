# OPIM 5604 - Predictive Analytics - Final Project: Predicting Student Dropout

*An .html version of this notebook is included if you do not wish to run the analysis on your machine.*

## Project Overview

This was a comprehensive work integrates the material learned in OPIM 5604 at the University of Commecticut (Predictive Analytics). The purpose of this project was to showcase my ability to execute the full predictive analytics process, including data transformation, exploratory analysis, and the application of machine learning algorithms in Python. The project highlights my analytical reasoning, statistical understanding, and proficiency in developing and evaluating predictive models. We had autonomy to discover a business problem, research a dataset, and propose actionable solutions from our findings.

The project focuses on predicting student dropout risk using demographic, academic, and socioeconomic variables. The goal is to create a model that identifies at-risk students for institutional intervention and to understand which factors most influence dropout risk. This analysis includes:

- Preparing and transforming student data for predictive modeling
- Exploring relationships between student characteristics and dropout risk
- Splitting the dataset into training, validation, and test sets for model evaluation
- Applying multiple machine learning algorithms to predict whether a student will graduate or drop out
- Evaluating model performance using metrics such as Accuracy, Recall, Precision, F1-Score, ROC-AUC, and Lift
- Highlighting which factors are most influential in predicting student dropout for potential institutional interventions

## Data Source

The project uses the following dataset -

Source: UC Irvine Machine Learning Repository | [Predicting Student Dropout](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)

| Dataset    | Description                                                                                                         |
|------------------|------------------------------------------------------|
| `data.csv` | Comprehensive dataset of 36 features and 4424 student observations. Data sourced from Portugal over a 10 year span. |

*Note: Although other academic papers on this topic exist using this dataset, this analysis was conducted independently and does not reference any external sources or other studies.*

## Methodology

### 1. Data loading, preprocessing, EDA

- Converted numeric codes in categorical variables into descriptive categories for readability
    - Example: Marital Status coded as 1-6 was transformed into signle, married, widower, divorced, facto union, or legally separated
- Conducted exploratory data analysis (EDA) to:
    - Identify nationalities that the students are representative of
    - Understand continuous variable distribution
    - Explore variable correlations
    - Get an idea of hour our continuous variables relates to our target variable
- Performed additional pre-processing for dimensionality reduction
    - Dataset contains 18 categorical variables, many with 15 or more sub-categories
    - Sub-categories with similar distributions relative to the response varaible were combined to reduce dimensionality
- Performed outlier detection and handling using IQR method to remove the top 1st quantile and the 99th quantile as to remove only extreme values. Some observations were realistic enough to expect

### 2. Modeling / Evaluation

- Split the data into X, y by selecting our predictors and target variable. X and y were then split into 60 / 20 / 20 Train / Validation / Test split for model evaluation *(sklearn train_test_split)*
- The target variable made this a classification problem. The following modeling methods were used:
    - Logistic Regression
    - Logistic Regression (Using only significant features p<.05)
    - Logistic Regression (Using only significant features and inclusion of an interaction term)
    - Decision Tree
    - Boosted Tree
    - Random Forest
    - Neural Network
    - Discriminant Analysis
    - K-Nearest Neighbors
    - Naive Bayes 
- All models that had tuneable hyperparameters were run through a grid-search to evaluate the best hyperparameters
- All models were evaluated using Accuracy, Precision, Recall, F1-Score, AUC-ROC, and Lift and comparing each against the training,validation, and test sets to assess performance and monitor for overfitting
- Each applicable model that included *feature_importances_* was run to understand which features were most important across different models *Note: Permutation Importance may be run on models that didn't include a feature_importances_ command, however that was outside the scope of this project*

## Conclusions and Findings

### Best Model
The "Best Model" was evaluated by comparing the evaluation metrics both between models and across the training, validation, and test sets. First, models with metric scores that remained closer on all 3 sets were identified as being less overfit. A model that performed significantly better on the training set than the validation and test sets signifies that it is biased towards the training data and may not perform on unseen data outside the scope of this project. A model that performs evenly across the validation and test sets helps give us confidence that introduced to new data, our model should perform relatively similarly.

The second consideration was to the context of this problem. The goal of the problem was to create a model that identifies at-risk students and flag them for intervention, making Recall* particularly important since missing a student who may drop out is more costly than incorrectly flagging one who would have graduated regardless. However, a model with a perfect Recall score is not automatically ideal. Other metrics, such as Precision and F1-Score alongside Recall were used to ensure the model balances correctly.

With both of these in mind, the best model was the Naive Bayes model with Recall consistent between .75 - .77. While this model received a lower F1 Score than many of the other models, some others showed overfitting or were too heavy on Accuracy.

*Recall is the score identifying ratio of True Dropouts to True Dropouts + Missed Cases

### Conclusions and Recommendations
This project creates a model that could be used to mark students as being a dropout risk or not using socioeconomic, academic, and demographic features. *(For more information on all features that were available, please refer to the documentation on the dataset source reference.)* Through the modeling, certain features were identified as being commonly important in usefulness to predict whether a student was identified as a dropout or non-dropout. These included whether a student had their tuition fees up to date, curricular units that a student was either approved for, graded for, or evaluated for during their first and second semesters, admission grades, previous qualification grades, or their age at enrollment. 

For tuition fees we see that of the students whose tuition fees were not up to date, 87% of them dropped out compared to 25% who did have their tuition fees up to date. This signifies that students who do not make payments or are struggling to make payments likely feel pressure from economic conditions and are likely to drop out due to inability to pay or potentially from laziness. Institutions may be able to implement payment plans, tuition assistance programs for students who are falling behind on payments, coaching for students who are struggling to stay on track, or simply to research other reasons for non-payment. 

For Curricular Units, this signifies students that are struggling academically. Instituations may be able to implement student support or tutoring assistance for students who are struggling to pass classes as a way to offer extra help. Students who are approved for 3 or less total credits are significantly more likely to drop out than students who are approved for 4 or more. There is a sweet spot in this datapoint between 6 - 8 credits where the dropout rate is below 10%.

For admission grade and previous qualification grades, students who have lower grades coming into the institutions have higher dropout rates. Institutions can potentially look to require additional pre-requisites or pre-classes before starting their academic careers. Once students below a certain admission grade or previous qualification grade complete these courses and re-test, they'll have the appropriate baseline to succeed in higher education. Alternatively, institutions can look to raise the lower threshold for acceptance, where students performing under a certain threshold are not admitted. While the second suggestion may reduce dropout rate, it is a little less inclusive and limits opportunity.

## Technology, Tools & Packages
- **Languages:** Python
- **Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn, statsmodels, scipy, geopandas, country-converter
- **Other Tools:** Google Colab, Jupyter Notebook, GitHub, VS Code

## How to Reproduce

An .html version of this notebook is included if you do not wish to run the analysis on your machine.

*This process uses the use of [UV](https://docs.astral.sh/uv/) to set up and run*

**Option 1: Using VS Code**
1. Clone the repository:  
   ```
   git clone https://github.com/schgrz/predicting-student-dropout.git
   ```
2. Open the project folder in VS Code. Make sure you have the Python and Jupyter Extensions installed. These can be found on the VS Code Extensions Marketplace.
3. **Initialize a virtual environment** (Python 3.12) *Note: Geopandas country visualization may not work above 3.12 at the time of publising*
   ```
   uv .venv

   source .venv/bin/activate # Mac/Linux
   .venv\Scripts\activate # Windows
   ```

   OR

   ```
   python -m venv .venv
   
   source .venv/bin/activate # Mac/Linux
   .venv\Scripts\activate # Windows
   ```
4. **Install dependencies**
   ```
   uv sync
   ```
5. **Verify folder structure**
   ```
   LinkedIn Sales Navigator (Working Directory)
   ├── data/                     # contains raw and processed datasets
   └── OPIM 5604 - Final Notebook.ipynb
   ```
6. **Run the notebook**
    1. Open `OPIM 5604 - Final Notebook.ipynb`  
    2. Execute all cells to reproduce the analysis, modeling, and visualizations.

**Option 2: Using Anaconda**
1. Clone the repository:  
   ```
   git clone https://github.com/schgrz/predicting-student-dropout.git
   ```
2. **Launch the project**
   1. Open Anaconda Navigator and Launch Jupyter Notebook
   2. Navigate to the project folder
   3. Open `OPIM 5604 - Final Notebook.ipynb`
3. **Install dependencies**
   ```
   !pip install pandas numpy scikit-learn matplotlib seaborn geopandas country-converter scipy statsmodels
   ```
4. **Run the notebook**

    Execute all cells to reproduce the analysis, modeling, and visualizations.

## Contact
- Email: alex.schwarzgrzesiak@gmail.com
- LinkedIn: [linkedin.com/in/schwarzalex](https://www.linkedin.com/in/schwarzalex/)
- Portfolio: [github.com/schgrz](https://github.com/schgrz)