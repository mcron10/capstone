
# Predicting Customer Satisfaction Using TripAdvisor Hotel Reviews

## Author
Mark Cronin

## Executive Summary

This project explores how the textual content of TripAdvisor hotel reviews can predict a customer's overall satisfaction rating (originally from 1 to 5). By processing the review text, extracting features using TF-IDF, and addressing class imbalances through oversampling, the aim is to identify which words, themes, and sentiments correlate strongly with particular ratings.

Our results show that traditional machine learning models (e.g., Logistic Regression, Naive Bayes, Random Forest) perform reasonably well, especially when the rating scale is simplified from five classes to three classes (e.g., mapping 4 → 5 and 2 → 1), resulting in a more balanced classification task. However, when retaining all five rating categories, models encounter more difficulties due to class imbalance and subtle differences between intermediate ratings. This suggests that while our current approach provides insights and a helpful starting point, future enhancements or more sophisticated NLP models might further improve accuracy and interpretability.

## Rationale

In the hospitality industry, understanding the key factors that lead to customer satisfaction is crucial. Online reviews influence booking decisions, and extracting actionable insights from these reviews can guide service improvements. This project’s data-driven approach enables hotel managers to focus on what matters most to guests, increasing the likelihood of positive experiences and favorable reviews.

## Research Question

Primary Question:
Can the textual content of TripAdvisor hotel reviews accurately predict a customer’s overall satisfaction rating, and can simplifying the rating scale improve model performance?

## Data Sources

TripAdvisor Hotel Reviews:
A dataset of textual reviews paired with numerical ratings (1 to 5). This dataset offers a range of customer feedback and sentiment.

No additional external datasets were used in this analysis.

## Methodology

### Preprocessing & Feature Engineering:
There was not missing data.
Clean and normalize text (tokenization, removal of punctuation, stop words, and domain-specific terms).
Lemmatize words to standardize forms.
Spell Check words.
Convert processed text into numerical feature vectors using TF-IDF.
Incorporate numeric features (word_count, text_length) if beneficial.

### Class Imbalance Handling:
Experiment with adjusting the rating scale from 5 classes (1, 2, 3, 4, 5) to 3 classes (1, 3, 5) by mapping 4→5 and 2→1.
Use oversampling techniques (e.g., ADASYN, SMOTE) to balance classes and improve model training stability.

### Models & Hyperparameter Tuning:
Evaluate multiple machine learning models (Naive Bayes, Logistic Regression, Random Forest, K-Nearest Neighbors).
Use GridSearchCV for hyperparameter tuning with 3-fold cross-validation.
Assess models using accuracy, precision, recall, F1-score, and training time.

### Validation with Synthetic Data:
Test models on synthetic reviews (good, neutral, bad) to ensure that predictions align with intuitive expectations.

### Results
The following results were taken with the spellchecker on and all stop words enabled.

#### Model Comparison: 5 Class Rating
Logistic Regression stands out for accuracy, while Naive Bayes and Random Forest offer more efficient training times at a slight cost to performance. KNN, however, does not compare favorably to the other approaches in this scenario.

                         Model  Accuracy    Time (s)  \
        0          Naive Bayes  0.554947   14.989442   
        1  Logistic Regression  0.592776  854.818252   
        2        Random Forest  0.563163  296.079400   
        3  K-Nearest Neighbors  0.363745   23.0

#### Model Comparison: 3 Class Rating
Under the 3-class rating system, Logistic Regression leads in accuracy, while Naive Bayes provides a near-competitive performance in a fraction of the training time. Random Forest also remains a solid middle-ground choice. KNN, though improved, still lags behind. Gradient Boosting is powerful but demands extensive computation, making it a potential choice only if the highest accuracy is absolutely necessary and resource constraints are minimal.

                         Model  Accuracy     Time (s)  \
        0          Naive Bayes  0.826772    11.743616   
        1  Logistic Regression  0.848853   179.576612   
        2        Random Forest  0.825745   186.171369   
        3  K-Nearest Neighbors  0.687265    18.697746   
        4    Gradient Boosting  0.838583  4169.587689   

### Best Performance on Three-Class System:
After reducing the rating categories to (1, 3, 5), the models, particularly Logistic Regression, improved in accuracy and provided more stable performance metrics.

### Challenges with All Five Classes:
Retaining the full 1-to-5 rating scale introduced complexity. Models struggled with intermediate classes due to subtle differences and class imbalance.

### Insights & Actionable Items:
The analysis highlights key words and themes associated with high satisfaction (e.g., "clean", "friendly") and dissatisfaction (e.g., "dirty", "rude"). Such insights can guide hotels to prioritize improvements.

## Future Work

### Advanced Models (CNNs, BERT):
Although not explored in this notebook, future research could involve more advanced NLP models, such as Convolutional Neural Networks (CNNs) or transformer-based models like BERT. These models often handle complex linguistic patterns more effectively and may yield better results, especially when dealing with the full 5-class rating system.

### Additional Features & Data:
Integrating other data (e.g., hotel metadata, reviewer profiles) or experimenting with different text representations and embeddings could further enhance performance and insights.

## Project Organization

### Notebook:
The main analysis notebook (capstone-module-final.ipynb) includes all preprocessing, modeling, evaluation, and visualizations.

### Repository Structure:
        README.md (this file)
        data/ (contains the dataset)
        notebooks/ (the Jupyter notebook)
        models/ (saved trained models)

The repository avoids unnecessary files and uses descriptive directory and file names.

## How to Run

Clone the repository and navigate to the project directory.
Install dependencies:

        pip install numpy pandas nltk textblob symspellpy pycountry scikit-learn imbalanced-learn matplotlib seaborn pandarallel spellchecker

Launch Jupyter Notebook:

        jupyter notebook

Open capstone-module-final.ipynb and run cells in order.

## Contact

## Links
Notebook :- capstone-module-final.ipynb

can be found here

https://github.com/mcron10/capstone.git

## Contact and Further Information
For further inquiries, please reach out at mcron10@wgu.edu





