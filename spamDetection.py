import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st 

#Multinomial NB classifies if text is spam or not

data = pd.read_csv('spam.csv')

data.drop_duplicates(inplace=True) 
#print(data.isnull().sum())

# output- category, input - message 
mess = data['Message']
cat = data['Category'] 

#splitting the data into training and testing sets
# 80% training, 20% testing 
(mess_train, mess_test, cat_train, cat_test) = train_test_split(mess, cat, test_size=0.2)

# now we'll convert the text data into numerical data using Countvectorizer. 
cv = CountVectorizer(stop_words='english')

# fit the training data i.e to transform it into numerical data
features = cv.fit_transform(mess_train) 

#creating model 
model = MultinomialNB() 

# training the model
model.fit(features, cat_train)  

#testing the model
features_test = cv.transform(mess_test) 
#print(model.score(features_test, cat_test))    # Output: 0.9857142857142858 (accuracy)

#predict data
def predict_spam(message):
    input_data = cv.transform([message]).toarray()
    prediction = model.predict(input_data)
    return prediction

# print(predict_spam('Congratulations! You have won a lottery worth ₹100000.'))  # Output: ['spam']
#building web application using streamlit
st.header('Spam Detection App')
input_msg = st.text_input("Enter your message:")

if st.button('Predict'):
    if input_msg:
        result = predict_spam(input_msg)
        if result[0] == 'spam':
            st.error("This message is spam.")
        else:
            st.success("This message is not spam.")
    else:
        st.warning("Please enter a message to predict.")