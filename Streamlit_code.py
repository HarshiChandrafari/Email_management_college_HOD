import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AutoModel
from groq import Groq
import streamlit as st
# model = AutoModel.from_pretrained("Harshi2104/lora_model_for_email_response_HOD")

PATH = 'fine_tuned_bert_20epoch' #Change this according to your path
model1 = BertForSequenceClassification.from_pretrained(PATH) # Load the fine-tuned model

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # Load the BERT tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Check if GPU is available

KEY1 = "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"
KEY2 = "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"
KEY3 = "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"
client = Groq(api_key=KEY1) #get your API key from https://groq.io/
api_keys=[KEY2,KEY3] #get multiple API keys with different Email addresses, you will run into rate limits a lot so either slow down the requests or let lots of api keys handle the job
api_keyover=0
model2='llama3-8b-8192'



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
        outputs = model1(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids) # Forward pass
        logits = outputs.logits  # Get the logits

    predicted_class = torch.argmax(logits, dim=1).item()  # Get the index of the max logit
    return id2label[predicted_class] # Get the predicted class label



def get_prediction(text, model=model2):
    global i, api_keyover, client
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": """You are a helpful assistant that generates responses to emails
                    Sensitive Emails:
                    Emails related to corporate enquiries or sensitive information (confidential partnerships or
                    legal matters) are escalated to the respective HODs for manual response.
                    ○ General Information:
                    For non-sensitive emails that fall under public knowledge (academic schedules, research
                    access, department procedures), the system drafts a response by pulling data from the
                    university’s document repositories.
                    Example: If a student emails asking for a course syllabus, the system accesses the
                    department’s document repository, fetches the syllabus, drafts a response and sends it.
                    ○ Research Queries:
                    ■ If the system identifies an email related to shared research data, it verifies whether the
                    required information is available in the database. If so, it formulates an appropriate response
                    and replie
                    """
                },
                {
                    "role": "user",
                    "content": f"""Respond to email, if they are asking for document direct them to  https://drive.google.com/drive/folders/1R8hD0h9EPx8vuEdJx6s8wXpmiawC1mQM?usp=sharing
                    if the mail is sensitive contains business, partnership, contract, legal, sponsorship.,condidencial, non disclosure reply with 
                    Thanks for the mail. Since this is a sensitive mail, we have shared this to the HOD. Please wait for their manual response. "Thank you for reaching out and expressing your interest in collaborating. I’m glad to hear about your work and the potential to share knowledge on this important topic.
                    Email :
                    {text} 
                    Just the email body no extra text to be returned
                    """
                }
            ],
            temperature=0.7,
            top_p=1,
            stream=False,
            stop=None,
        )
    except Exception as e:
        print('API key limit reached', e)
        api_keyover += 1
        client = Groq(api_key=api_keys[api_keyover % 2])

    try:
        answer = completion.choices[0].message.content.strip()
        return answer
    except Exception as e:
        return e

    return None


st.title('Contact HOD') # Title of the web app

# Input text from the user
input_text = st.text_area("Enter the message:") # Text area for user input

if st.button('Send'):
    if input_text:
        label = predict(input_text,id2label) # Get the predicted class label
        st.write(f'The message has been categorized to: **{label}**'  ' category') # Display the predicted class label
        response = get_prediction(input_text,model2)
        if response == "cannot access local variable 'completion' where it is not associated with a value":
            response = "Wait for manual response system not working"
        st.write(f'{response}') # Display the predicted class label
    else:
        st.write("Please enter the message you want to send") # Display this message if the user has not entered any text

