import joblib
import re
import streamlit as st
from lime.lime_text import LimeTextExplainer
import streamlit.components.v1 as components
from transformers import DistilBertTokenizer
from transformers import TFDistilBertForSequenceClassification
import torch.nn.functional as F
import torch

import tensorflow as tf
import nltk
from nltk.corpus import stopwords
 
nltk.download('stopwords')
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()


st.write("# Customer Complaints Classifier")

complaint_text = st.text_input("Enter a complaint for classification")


def preprocessor(text):
    text = re.sub('<[^>]*>', '', text) # Effectively removes HTML markup tags
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text


#complaint_text = preprocessor(complaint_text)
#stop_words = set(stopwords.words('english'))
#word_tokens = w_tokenizer.tokenize(complaint_text)
#complaint_text = ' '.join([w for w in word_tokens if not w.lower() in stop_words])
#complaint_text = ' '.join([lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(complaint_text)])


model = TFDistilBertForSequenceClassification.from_pretrained("CustomModel2")

save_directory = r"D:\LEARNING\WELLS FARGO\NLP\CFPB\Jupyter notebooks\Transformers\Fine tuning Distil BERT\saved_models" 
tokenizer_fine_tuned = DistilBertTokenizer.from_pretrained(save_directory)


def classify_complaint(model, complaint):

    predict_input = tokenizer_fine_tuned.encode(
        complaint,
        truncation = True,
        padding = True,
        return_tensors = 'tf'    
        )
    output = model(predict_input).logits.numpy()[0]
    prediction_value = tf.argmax(output).numpy()
    label = prediction_value

    
    product_dict ={0:'credit_card',1:'credit_reporting',2:'debt_collection', 
                3:'mortgages_and_loans',4:'retail_banking'}
    torch_logits = torch.from_numpy(output)
    probabilities_scores = F.softmax(torch_logits).detach().numpy()
      
    return {'label': product_dict[label], 'complaint_prob': probabilities_scores[label]}



def predictor(text):
    predict_input = tokenizer_fine_tuned.encode(
        text,
        truncation = True,
        padding = True,
        return_tensors = 'tf'    
        )
    output = model(predict_input).logits.numpy()
    torch_logits = torch.from_numpy(output)
    probas = F.softmax(torch_logits).detach().numpy()
    return probas



if complaint_text != '':
    result = classify_complaint(model, complaint_text)
    st.write(result)
    
    explain_pred = st.button('Explain Predictions')
    if explain_pred:
        with st.spinner('Generating explanations'):
            class_names = ['credit_card','credit_reporting','debt_collection','mortgages_and_loans','retail_banking']
            explainer = LimeTextExplainer(class_names=class_names)
            exp = explainer.explain_instance(complaint_text, predictor, num_features=5, num_samples=1,top_labels=2)

            components.html(exp.as_html(), height=3500)