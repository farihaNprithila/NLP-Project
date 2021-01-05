import ktrain
from ktrain import text

# fetch the dataset using scikit-learn
categories = ['alt.atheism', 'soc.religion.christian',
              'comp.graphics', 'sci.med']
import pandas as pd

df = pd.read_csv('Reddit_Data.csv')
from sklearn.model_selection import train_test_split

# dropping the nan value
df = df.dropna()

# Mapping -1 and 1 to positive and negative for better readability and testing
df['category'] = df['category'].map({-1: "Negative", 1: "Positive", 0: "Neutral"})
print(df['category'])
# split into train data, validation data and test data
raw_train, test = train_test_split(df, test_size=0.2, random_state=2020)
train, val = train_test_split(raw_train, test_size=0.3, random_state=2020)

x_train = train['clean_comment'].tolist()
y_train = train['category'].to_numpy()
x_val = val['clean_comment'].tolist()
y_val = val['category'].to_numpy()

# Converting into bert tokenizer
(X_train, y_train), (X_val, y_val), preproc = text.texts_from_array(x_train=x_train, y_train=y_train,
                                                                    x_test=x_val, y_test=y_val,
                                                                    class_names=['Negative', 'Positive', 'Neutral'],
                                                                    preprocess_mode='bert',
                                                                    maxlen=500)

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
predictor.save('/Users/Prithila/Desktop/Project NLP/bertreddit3')