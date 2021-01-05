# Importing required libraries
import ktrain
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Loading the pretrained model
predictor = ktrain.load_predictor('/Users/Prithila/Desktop/Project NLP/bertreddit')

# Loading data from the csv
df = pd.read_csv('../Training/Reddit_Data.csv')

# Doing all the conversions again to get the same test data
df = df.dropna()
df = df[(df[['category']] != 0).all(axis=1)]
df['category'] = df['category'].map({-1: "Negative", 1: "Positive"})
raw_train, test = train_test_split(df, test_size=0.2, random_state=2020)
train, val = train_test_split(raw_train, test_size=0.3, random_state=2020)

# Converting the data to list for prediction
data = test['clean_comment'].tolist()
y_true = test['category'].tolist()

# Giving the test data to the model for prediction
y_pred = predictor.predict(data)

# Showing the results and accuracy of the model
print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
