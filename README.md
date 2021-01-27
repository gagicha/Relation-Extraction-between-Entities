# Relation Extraction between Entities(NLP)

This project named "Relation Extraction between two named Entities in a given sentence" is for my NLP course at UT Dallas. 

Given a sentence with 2 named entities enclosed in entity tags <e1>, <e2>, we have to find the relation between them and the direction in which the relationship holds. 
We are given a training and test data file with many such sentences. 

For example: 

" <e1> Leland High School </e1> is a public high school located in the Almaden Valley in <e2> San Jose </e2> California USA in the San Jose Unified School District . "
org:city_of_headquarters(e1,e2)
In the above sentence, Entity1 is Leland High School, Entity2 is San Jose, Relationship between the entities: org:city_of_headquarters, Direction of the relationship: (e1,e2)

The aim is to use basic Machine Learning techniques to extract meaningful features from the sentences, train ML model over these features and relationships extracted, and predict the correct relationship and the direction for the test dataset. 

Dataset:
Dataset has 17k+ training example and 3k+ testing example but has few missing entities. This dataset has relationships related to ORG, PER etc.

Dependencies:

pandas
numpy
spacy
nltk
networkx
sklearn
xgboost
gensim
category_encoders
pickle

-------------------------------------------------------------------------------------------------------------------------
Extract_features.ipynb

This file takes train.txt and test.txt as inputs and extracts all the features from it and saves them in train.csv and test.csv files. Features extracted are Part-of-speech(POS), ENR, Entities e1 and e2, shortest dependency path between entities, words in between, root words.

-------------------------------------------------------------------------------------------------------------------------
word_embeddings.ipynb 

This file reads the train and test csv files and trains the word2vec model from gensim over all the sentences. The model is then saved into'model.bin' file. The model transforms each word into 100 dimensional vector. 

-------------------------------------------------------------------------------------------------------------------------
encodings.ipynb

This file reads train.csv, test.csv files and word2vec trained model. It encodes pos_e1, pos_e2, enr_e1, enr_e2 using Binary Encoder and save the encodings into 'train_encodings_pos_enr.csv' and 'test_encodings_pos_enr.csv' files. It encodes entities using word2vec model and store them into 'train_encodings_e1_e2.csv' and 'test_encodings_e1_e2.csv' files. It encodes shortest dependency path and stores them into 'train_encodings_SDP.csv' and 'test_encodings_SDP.csv' files. Words in between are encoded and stored in 'train_enc_words_in_between.csv' and 'test_enc_words_in_between.csv' files. It encodes root words and store into 'train_enc_root.csv' and 'test_enc_root.csv' files. It encodes labels using label encoder and stores them into 'train_label_enc.csv' and 'test_label_enc.csv' files. Stored all the above into 'train_pos_enr_e1e2_root_between.csv''train_pos_enr_e1e2_root_between.csv' and 'test_pos_enr_e1e2_root_between.csv' files. 

-------------------------------------------------------------------------------------------------------------------------
model.ipynb

imports all the encoded files and runs Decision tree, XGBoost and SVM models on different combinations of features. A result file is generated where results for all features and models are saved. The best model is then saved for future prediction. 

-------------------------------------------------------------------------------------------------------------------------
demo.ipynb
This file takes a sample input and extractes all the features from it, encodes those features and run the best model earlier saved over this to get the model prediction.  

Results:
Our model reached f1-score of 0.50 while the best scores for this dataset is around 0.6

