import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the fine-tuned BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained('fine_tuned_bert') 
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define id2label mapping
id2label = {
    1: 'Student',
    2: 'Research',
    0: 'Corporate',
    3: 'Sensitive/Confidential'
}

def predict(text,id2label):
    # Preprocess the input text
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        return_token_type_ids=True,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',  # Return PyTorch tensors
    )

    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    token_type_ids = encoding['token_type_ids']

    with torch.no_grad(): 
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)
        logits = outputs.logits  

    predicted_class = torch.argmax(logits, dim=1).item()  # Get the index of the max logit
    return id2label[predicted_class]



st.title('Contact HOD')

# Input text from the user
input_text = st.text_area("Enter the message:")

if st.button('Send'):
    if input_text:
        # Get prediction and display it
        label = predict(input_text,id2label)
        st.write(f'The predicted label is: **{label}**')
    else:
        st.write("Please enter some text for classification.")
