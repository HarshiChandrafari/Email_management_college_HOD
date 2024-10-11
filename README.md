# Instructions to use


# Model Descriptions

## Model From Scratch 
To train this model, we began by cleaning the dataset. First, we removed all links and irrelevant data. After that, we tokenized the text, eliminated stop words, and lemmatized the words to their base forms, retaining only nouns, adjectives, adverbs, and verbs. The vocabulary was then encoded using Keras' TextVectorization layer.

The current model architecture consists of two Bidirectional LSTM layers: the first with 64 units and the second with 32 units. These are followed by a dense layer with 64 units and an output layer with 4 units, corresponding to the 4 target categories. We used ReLU activation in both dense layers.

LSTMs excel at capturing long-range dependencies and relationships between words in a sentence, making them ideal for natural language processing tasks. This is why we have used LSTMs in our model. 

We trained the model for 25 epochs. Although we considered increasing the number of epochs, we observed signs of overfitting, so we settled on 25.

We also experimented with modifying the model architecture by adjusting the number of layers and units. Reducing these resulted in slight drops in recall and precision, while increasing them didn't yield significant improvements. Based on these trials, we chose the structure described above as the most effective. The metrics obtained by different model architecture can be viewed in the link provided below:

https://docs.google.com/document/d/1G1hHA1_NriG2QRxwFJPNPHn4YLxn40TthS5v6tkflC0/edit

Finally, we obtained the following results:
For test data:
Precision: 0.82
Recall: 0.802

For train data
Precision: 0.96
Recall: 0.96

[https://docs.google.com/document/d/1PyQOHmqc3Q44WixoPSKDxTGdNp-fj0uIi2dGOX-K3Yw/edit?usp=sharing]

## Model Fine-Tuned

Link to saved model on drive -> [https://drive.google.com/drive/folders/1-SJ2l5rR-OtguvAPZT8ekN5bio23HGzW?usp=sharing]
