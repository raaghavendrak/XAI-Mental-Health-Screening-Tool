import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import ollama
import pickle

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from lime.lime_text import LimeTextExplainer

max_length = 200   # Max length of each sequence
trunc_type = 'post' # Chop off the end if longer than max_length
padding_type = 'post' # Add zeros at the end if shorter

# 1. Load the trained model and tokenizer
model = load_model(r'..\Model\BiLSTM\model_bilstm.keras')

# 2. Load the tokenizer (ensure this matches your training stage)
with open(r'..\Model\BiLSTM\tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def predict_risk(text):
    # Preprocess the input text
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=max_length, 
                           padding=padding_type, truncating=trunc_type)
    
    # Perform inference
    prediction = model.predict(padded)
    
    # For Binary Classification (0 or 1)
    # If using a sigmoid activation in the final layer:
    score = prediction[0][0]
    risk_class = 1 if score > 0.5 else 0
    
    # This call runs entirely on your CPU
    response = ollama.chat(model='phi4-mini', messages=[
        {'role': 'user', 'content': text},
    ])

    return risk_class, score, response['message']['content']

#----------------------------------------------------------------------------------
# 1. Define the prediction wrapper
def predict_probs(texts):
    # Convert raw text strings to padded sequences
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_length)
    
    # Get model predictions
    preds = model.predict(padded)
    
    # BiLSTM output is usually (batch, 1) for sigmoid. 
    # LIME needs (batch, 2) for [class_0_prob, class_1_prob]
    if preds.shape[1] == 1:
        probs = np.hstack([1 - preds, preds])
        return probs
    return preds

# 2. Initialize the explainer
explainer = LimeTextExplainer(class_names=['Non-Suicide', 'Suicide'])

st.title("Explainable AI (XAI) Mental Health Screening Tool")
comment = st.text_area("***Enter Comment :***")
if st.button("Analyze") and comment:
    label, confidence, llmResp = predict_risk(comment)
    st.write(f"Predicted Class: {label} (Confidence: {confidence:.4f})")
    st.markdown(llmResp)

    # 1. Generate the explanation
    exp = explainer.explain_instance(comment, predict_probs, num_features=10)

    # 2. Convert the explanation to an HTML string
    # We use .as_html() instead of .show_in_notebook()
    obj_html = exp.as_html()

    # 3. Render it in Streamlit
    st.subheader("Model Explanation")
    st.iframe(obj_html, height=800)

#whats the point of even trying i legit fail at everything i try to do i cannot hold down a job i fantasize about hitchhiking and living off the road but do not have the courage to give up my shitty life to do it and i mean shitty my and my gf fight nearly everyday about the most pointless bullshit i lost my job today another in the long stream of failures that is my life just when i think i am figuring it out and doing good some shit ha to go wrong
