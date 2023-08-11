# News-category-classification
News category classification using bag of n-grams text representation

# Importing Libraries:
The necessary libraries are imported, including spacy for natural language processing, pandas for data manipulation, and various classes and functions from scikit-learn for machine learning tasks.

# Loading Language Model:
The code loads a pre-trained English language model from spaCy's "en_core_web_sm".

# Preprocessing Function (preprocess):
A function named preprocess is defined. This function takes a text as input and processes it using the loaded spaCy model. It tokenizes the text, removes stopwords and punctuation, and lemmatizes the remaining tokens. The filtered tokens are then joined back into a string and returned.

# Loading and Preparing Data:
The code reads data from a JSON file named "news_dataset.json" and creates a pandas DataFrame named df. The shape of the DataFrame (number of rows and columns) is printed, and the first few rows of data are displayed.

# Balancing the Dataset:
To balance the dataset, the code takes a fixed number of samples from each category (BUSINESS, SPORTS, CRIME, and SCIENCE) using the .sample() method. These subsets are then concatenated vertically to create a new DataFrame named df_balanced, which contains an equal number of samples from each category.

# Mapping Categories to Numbers:
A mapping dictionary named target is created to associate category names with numeric labels. A new column named 'category_num' is added to the df_balanced DataFrame, where category names are replaced with their corresponding numeric labels using the .map() method.

# Splitting Data into Train and Test Sets:
The text data from the 'text' column of the df_balanced DataFrame is preprocessed using the preprocess function. The data is then split into training and testing sets using train_test_split(). The training features (X_train) and labels (y_train) are set, as well as the testing features (X_test) and labels (y_test). The test set size is defined as 20% of the data, and stratified sampling is used to ensure a balanced distribution of categories in both sets.

# Creating a Pipeline:
A machine learning pipeline named clf is created using scikit-learn's Pipeline class. The pipeline consists of two stages:

The vectorizer_bow stage uses a bag-of-words vectorizer with an n-gram range of (1, 2) to convert the preprocessed text into a numerical feature matrix.
The Multi NB stage utilizes a Multinomial Naive Bayes classifier for classification.

# Fitting the Model:
The pipeline (clf) is trained using the training data (X_train and y_train) by calling the .fit() method.

# Making Predictions:
The pipeline predicts the categories for the test set (X_test) and stores the predictions in the y_pred variable.

# Printing Classification Report:
The classification_report() function is used to generate a comprehensive text report that presents various classification metrics, such as precision, recall, F1-score, and support, for each category.

# Confusion Matrix:
A confusion matrix is calculated using the confusion_matrix() function based on the predicted labels (y_pred) and true labels (y_test).

# Visualizing Confusion Matrix:
The confusion matrix is visualized using a heatmap created with matplotlib and seaborn. This heatmap helps to better understand the distribution of predicted and true labels, offering insights into the model's performance for each category.

In summary, the code takes raw text data, preprocesses it, balances the dataset, builds a machine learning pipeline, trains a classifier, evaluates its performance, and provides visualizations to aid in understanding the classification results.
