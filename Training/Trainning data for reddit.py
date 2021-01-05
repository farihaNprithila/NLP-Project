# Importing the required library
import pandas as pd
import ktrain
from ktrain import text
from sklearn.model_selection import train_test_split

# Pandas dataframe to read the data from csv file
df = pd.read_csv('Reddit_Data.csv')

# dropping the nan value
df = df.dropna()

# For initial case, just positive and negative sentiment has been taken, so dropping the neutral texts
# df = df[(df[['category']] != 0).all(axis=1)]

# Mapping -1 and 1 to positive and negative for better readability and testing
df['category'] = df['category'].map({-1: "Negative", 1: "Positive"})

# split into train data, validation data and test data
raw_train, test = train_test_split(df, test_size=0.2, random_state=2020)
train, val = train_test_split(raw_train, test_size=0.3, random_state=2020)

# Pre processing the train and validation set for pre-trained BERT model so that all the tokenization
# will be done according to BERT model
# maxlen is the length of the text
(X_train, y_train), (X_val, y_val), preproc = text.texts_from_df(train_df=train,
                                                                 text_column='clean_comment',
                                                                 label_columns='category',
                                                                 val_df=val,
                                                                 maxlen=500,
                                                                 preprocess_mode='bert')

# Using the BERT model as classifier
model = text.text_classifier(name='bert',
                             train_data=(X_train, y_train),
                             preproc=preproc)

# Preparing the optimal learner for the training
learner = ktrain.get_learner(model=model, train_data=(X_train, y_train),
                             val_data=(X_val, y_val),
                             batch_size=6)

# Finding which learning rate will be good for the model
learner.lr_find()

# Fitting the learning rate for next training and declaring the epochs
learner.fit_onecycle(lr=2e-5, epochs=5)

# Saving the best model for testing
predictor = ktrain.get_predictor(learner.model, preproc)
predictor.save('/Users/Prithila/Desktop/Project NLP/bertreddit')
