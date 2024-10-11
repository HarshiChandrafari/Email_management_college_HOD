# Instructions to run the streamlit app

* To run the streamlit application, make sure all dependencies in requirments.txt are installed
* Next, Download the model in the same directory where you have cloned the repository (i.e, inside the SmartSense_email_management folder]
* Link to download - https://drive.google.com/drive/folders/1DVBvNmdeYemHvt0Acg4BFl9XKaSUeDgx?usp=sharing
* After Downloading, run the streamlit file by typing - streamlit run Streamlit_code.py
* The app will open in the browser; input the email text that you want to classify.

# Data Collection

# Model Descriptions

## Model From Scratch for classification
To train this model, we began by cleaning the dataset. First, we removed all links and irrelevant data. After that, we tokenized the text, eliminated stop words, and lemmatized the words to their base forms, retaining only nouns, adjectives, adverbs, and verbs. The vocabulary was then encoded using Keras' TextVectorization layer.

The current model architecture consists of two Bidirectional LSTM layers: the first with 64 units and the second with 32 units. These are followed by a dense layer with 64 units and an output layer with 4 units, corresponding to the 4 target categories. We used ReLU activation in both dense layers.

LSTMs excel at capturing long-range dependencies and relationships between words in a sentence, making them ideal for natural language processing tasks. This is why we have used LSTMs in our model. 

We trained the model for 25 epochs. Although we considered increasing the number of epochs, we observed signs of overfitting, so we settled on 25.

We also experimented with modifying the model architecture by adjusting the number of layers and units. Reducing these resulted in slight drops in recall and precision while increasing them didn't yield significant improvements. Based on these trials, we chose the structure described above as the most effective. The metrics obtained by different model architectures can be viewed in the link provided below:

[https://docs.google.com/document/d/1PyQOHmqc3Q44WixoPSKDxTGdNp-fj0uIi2dGOX-K3Yw/edit?usp=sharing]

Final Results:

For test data:
* Precision: 0.82
* Recall: 0.802

For train data:
* Precision: 0.96
* Recall: 0.96


## Model Fine-Tuned for classification
The data pre-processing was the same as for the model made from scratch. The data was then tokenized using BertTokenizerFast.from_pretrained('bert-base-uncased'). Since BERT has a fixed input size, we set a maximum sequence length (`MAX_LEN`) of 512 tokens to ensure that our input data fits within the model's constraints.
In addition to tokenization, we created attention masks to differentiate between actual tokens and padding tokens, enhancing the model's focus during training. 

We used the BERT model for sequence classification tasks. The model, instantiated using `BertForSequenceClassification,` is specifically designed to handle classification with multiple labels (in our case, four).
The reason for choosing the BERT model for the fine-tuning model is that it is an effective model for text classification due to its bidirectional processing, which captures context from both sides of a word, enhancing semantic understanding. Its pre-training on large datasets allows for fine-tuning specific tasks with smaller datasets, and in our case, we had a small dataset.

We initially trained for 10 epochs, and the results were not satisfactory; therefore, we increased the number of epochs to 20

Final Results:

For Test Data
* Precision: 0.847
* Recall: 0.848

For train data:
* Precision: 0.94
* Recall: 0.95

The savel models, after fine-tuning, were too big and could not be uploaded to GitHub; therefore, we uploaded them to Google Drive and made them publicly available; you can download them and use them.
Model trained for 20 Epochs(Final Model) - [https://drive.google.com/drive/folders/1DVBvNmdeYemHvt0Acg4BFl9XKaSUeDgx?usp=sharing]
Model trained for 10 Epochs(Initial Model) - [https://drive.google.com/drive/folders/1-SJ2l5rR-OtguvAPZT8ekN5bio23HGzW?usp=sharing]

## Model for generating Email response

We fine-tuned the base model-"unsloth/llama-3-8b-Instruct-bnb-4bit" on responses.csv dataset, to generate the appropriate response.

Link to dataset on hugging face - https://huggingface.co/datasets/Harshi2104/Fine_tune_llama_email_response

Link to saved model on hugging face - https://huggingface.co/Harshi2104/lora_model_for_email_response_HOD

Colab File used for fine tuning - https://colab.research.google.com/drive/1hLSbmPFqxpYlqK5dqdmGtqU3UACHkgj5?usp=sharing

