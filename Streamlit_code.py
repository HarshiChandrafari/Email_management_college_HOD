import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AutoModel
# model = AutoModel.from_pretrained("Harshi2104/lora_model_for_email_response_HOD")

PATH = 'fine_tuned_bert_20epoch' #Change this according to your path
model = BertForSequenceClassification.from_pretrained(PATH) # Load the fine-tuned model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # Load the BERT tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Check if GPU is available

id2label = {
    1: 'Student',
    2: 'Research',
    0: 'Corporate',
    3: 'Sensitive/Confidential'
} # Mapping of class indices to class labels

def predict(text,id2label):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        return_token_type_ids=True,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',  # Return PyTorch tensors
    ) # Tokenize the input text

    input_ids = encoding['input_ids'] # Get the input ids
    attention_mask = encoding['attention_mask'] # Get the attention mask
    token_type_ids = encoding['token_type_ids'] # Get the token type ids

    with torch.no_grad(): 
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids) # Forward pass
        logits = outputs.logits  # Get the logits

    predicted_class = torch.argmax(logits, dim=1).item()  # Get the index of the max logit
    return id2label[predicted_class] # Get the predicted class label


st.title('Contact HOD') # Title of the web app

# Input text from the user
input_text = st.text_area("Enter the message:") # Text area for user input

if st.button('Send'):
    if input_text:
        label = predict(input_text,id2label) # Get the predicted class label
        st.write(f'The message has been categorized to: **{label}**' 'category') # Display the predicted class label
    else:
        st.write("Please enter the message you want to send") # Display this message if the user has not entered any text