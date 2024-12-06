
# Predicting Customer Satisfaction Using TripAdvisor Hotel Reviews

## Author
Mark Cronin

## Executive Summary
This project explores how the sentiment, themes, and specific content within TripAdvisor hotel reviews can predict a customer’s overall satisfaction rating. Using Natural Language Processing (NLP) and machine learning, we analyze textual reviews to uncover patterns and correlations with ratings, aiming to create a model that supports hotel managers in prioritizing improvements and addressing common issues effectively.

## Rationale
Understanding which aspects of a hotel experience resonate positively or negatively with customers is crucial in the competitive hospitality industry. Customer reviews significantly influence booking decisions, and hotels must identify actionable insights from these reviews. By systematically analyzing review content, this project provides a measurable framework to enhance guest experiences.

## Research Question
Can the sentiment, themes, and specific content within TripAdvisor hotel reviews accurately predict the customer’s overall rating?

## Data Sources
The primary dataset comprises hotel reviews and ratings from TripAdvisor. This dataset includes thousands of reviews, offering a broad spectrum of customer feedback. Supplementary data on hotel attributes or reviewer profiles may also be considered if necessary.

## Methodology
1. **Data Adjustment:**
   - Initial results with all five rating categories (1, 2, 3, 4, 5) were poor due to class imbalance.
   - Ratings were consolidated: rating `4` was converted to `5` and rating `2` was converted to `1`, effectively reducing the problem to three classes (1, 3, 5) for better performance.

2. **Natural Language Processing (NLP):**
   - Preprocessing: Tokenization, stop word removal, lemmatization, and TF-IDF feature extraction.
   - Sentiment and theme analysis to identify patterns indicative of satisfaction levels.

3. **Machine Learning Models:**
   - Models evaluated include Naive Bayes, Logistic Regression, Random Forests, and K-Nearest Neighbors.
   - Grid search and cross-validation were used for hyperparameter tuning.

4. **Feature Engineering:**
   - Extracted top words and themes for each rating.
   - Balanced class distributions using the ADASYN oversampling technique.

5. **Evaluation Metrics:**
   - Accuracy, F1 score, and training time were used to assess model performance.

## Results

### Model Comparison:
| Model                 | Accuracy | Time (s)   | Best Hyperparameters                                   |
|-----------------------|----------|------------|-------------------------------------------------------|
| Naive Bayes           | 0.752653 | 1.800694   | {'alpha': 10.0}                                      |
| Logistic Regression   | 0.814447 | 1.879964   | {'C': 0.01}                                          |
| Random Forest         | 0.806060 | 122.370477 | {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 50} |
| K-Nearest Neighbors   | 0.360664 | 129.682749 | {'metric': 'euclidean', 'n_neighbors': 3, 'weights': 'distance'} |

### Detailed Classification Reports:

#### Naive Bayes:
- **Time Taken:** 1.80 seconds
- **Accuracy:** 75.26%
- **Classification Report:**
  ```
              precision    recall  f1-score   support

           1       0.62      0.63      0.63       901
           3       0.21      0.24      0.22       616
           5       0.87      0.85      0.86      4325

    accuracy                           0.75      5842
   macro avg       0.57      0.57      0.57      5842
weighted avg       0.76      0.75      0.76      5842

#### Logistic Regression:
- **Time Taken:** 1.88 seconds
- **Accuracy:** 81.44%
- **Classification Report:**
  ```
              precision    recall  f1-score   support

           1       0.74      0.67      0.71       901
           3       0.30      0.29      0.29       616
           5       0.89      0.92      0.91      4325

    accuracy                           0.81      5842
   macro avg       0.65      0.63      0.64      5842
weighted avg       0.81      0.81      0.81      5842
  ``
#### Random Forest:
- **Time Taken:** 122.37 seconds
- **Accuracy:** 80.60%
- **Classification Report:**
  ```
              precision    recall  f1-score   support

           1       0.83      0.46      0.59       901
           3       0.38      0.01      0.02       616
           5       0.81      0.99      0.89      4325

    accuracy                           0.81      5842
   macro avg       0.67      0.49      0.50      5842
weighted avg       0.76      0.81      0.75      5842
  ``
#### K-Nearest Neighbors:
- **Time Taken:** 129.68 seconds
- **Accuracy:** 36.06%
- **Classification Report:**
  ```
              precision    recall  f1-score   support

           1       0.26      0.30      0.28       901
           3       0.13      0.65      0.21       616
           5       0.87      0.33      0.48      4325

    accuracy                           0.36      5842
   macro avg       0.42      0.43      0.32      5842
weighted avg       0.70      0.36      0.42      5842

### Installation Instructions

Before running the code, ensure all required libraries are installed. Below is a categorized list of libraries commonly used for data analytics and machine learning projects, along with installation instructions.
Common Libraries for Data Analytics and Machine Learning

#### Standard Libraries (Included in Python Standard Library):
        time
        re

#### Numerical and Data Manipulation Libraries:
        numpy
        pandas

#### Natural Language Processing Libraries:
        nltk
        textblob
        spellchecker
        symspellpy
        pycountry

#### Machine Learning Libraries:
        scikit-learn (sklearn)
        Additional Models: GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, KNeighborsClassifier, MLPClassifier, LinearDiscriminantAnalysis

#### Oversampling Libraries:
        imbalanced-learn (imblearn)

#### Visualization Libraries:
        matplotlib
        seaborn

#### Parallel Processing Libraries:
        pandarallel

#### Other Libraries:
        pkg_resources
        collections.Counter
        
### Installation Methods
#### Using pip (Preferred for Most Libraries)

Run the following commands in your terminal or Jupyter Notebook to install the required libraries:

pip install numpy pandas nltk textblob symspellpy pycountry scikit-learn imbalanced-learn matplotlib seaborn pandarallel spellchecker

#### Using conda (For Anaconda Users)

If you use the Anaconda distribution, you can install most of the libraries using conda:

conda install numpy pandas nltk scikit-learn matplotlib seaborn
conda install -c conda-forge textblob symspellpy pycountry pandarallel spellchecker imbalanced-learn

## Next Steps
The preferred model for this project is Logistic Regression due to its highest accuracy (81.4%) and strong performance across all classes, as shown in the classification reports. It balances precision and recall effectively and is computationally efficient compared to other models like Random Forest and K-Nearest Neighbors. Random Forest, while competitive in accuracy, struggled with recall for specific classes and had significantly longer training times, making it less practical.

Next steps for the final capstone submission include finalizing Logistic Regression as the selected model and documenting its hyperparameters ({'C': 0.01}). Additional visualizations, such as confusion matrices and ROC curves, will be included to support the results. The analysis should also highlight key features influencing predictions, such as significant review themes or sentiments. The README.md will be updated with these findings and links to the final Jupyter Notebooks, emphasizing actionable insights for non-technical audiences.

To ensure completeness, the project will meet all rubric criteria, including clean code, organized documentation, and visualizations. The GitHub repository will include structured folders, the final model, and relevant scripts. Finally, cross-validation and testing on unseen data may be performed to confirm stability and generalizability. These steps will ensure a polished and comprehensive capstone submission.

## Links
1. **Notebook :- capstone-module-20.ipynb **
   - can be found here
   - https://github.com/mcron10/capstone.git

## Contact and Further Information
For further inquiries, please reach out at mcron10@wgu.edu
