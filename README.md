Project 3
============

Matthew Huson
--------------
cs5293, Spring 2020
-------------------



### PYTHON VERSION USED ###

**Python 3.6.8**

### LIBRARIES NEEDED ###


* pandas - external
* numpy - external
* sklearn - external
* gensim - external - only used for data exploration, not in final code
* spacy - external - only used for data exploration, not in final code

### EXECUTION ###

With the cs5293-project3 folder open, this program can be accessed by entering the command

~~~
jupyter notebook
~~~

in the terminal, and then selecting project3.ipynb from the menu found there. Then, just work through the notebook

### PROMPT QUESTIONS ###

#### How did you turn your text into features and why? ####

Many different inputs were attempted for this project. 
To begin with, two ingredients lists were obtained: "cleaned" and "uncleaned". 
The cleaned ingredient list was put through some basic pre-processing. 
The uncleaned was left as-is (for some vectorizers the individual words were joined together, but the content was unchanged). 
Then each list was vectorized three different ways: Tfidf, Word2Vec and spaCy/gloVe. 
The result was 6 sets of vectors: un/cleaned Tfidf, un/cleaned Word2Vec and un/cleaned spaCy/gloVe. 
Each of the 6 sets of vectors were evaluated on a number of different models, using cross_val_score from sklearn. 
Surprisingly, it was found that the uncleaned Tfidf tended to return the best accuracy score, regardless of classifier. 
This, combined with the fact that it runs pretty quickly is why it was chosen as the text to features function for this project. 

#### What classifiers/clustering methods did you choose and why? ####

There were 4 different classifiers tested for this project, and a number of hyperparameter combinations. 
The classifiers are: k-nearest neighbors, linear support vector classifier, logistic regression and extremely randomized trees. 
For logreg and linearSVC the hyperparameters were a C-value of 1 and 0.5. 
For kNN the hyperparameters were 5 and 10 nearest neighbors. 
For extremely random trees the hyperparameters were 100, 200 and 400 estimators. 
Each of the 9 classifier/hyperparameter combinations were tested on all of the data inputs. 
LinearSVC and logistic regression both performed the best (both on Tfidf data). 
k-nearest neighbors was within 0.04 accuracy of the first two, trains much more quickly, and has a method for finding nearby recipes. 
Because of this, it was chosen over the better performing models, since this is to be a useable app, rather than a competition submission. 
kNN with 11 neighbors performed slightly better than 10, so 11 was the hyperparameter chosen. 

#### What N did you choose and why? ####

Since a hotel is open 7 days a week, it makes sense to have 7 similar recipes. 
That way, there can be a different meal every day, giving the illusion of variety. 
It also appears that 7 is a number that returns recipes within a Euclidean distance of 1-2. 
7 is also a good number to fit on a computer output. 

### ASSUMPTIONS/BUGS ###

* while linearSVC was found to have the best cross validated accuracy score, k-nearest neighbors was chosen as the classifier due to computational demands
* because min_df is set to 2 in the vectorizer, if the user inputs an ingredient that only shows up once in the original dataframe, the vectors and matrices will have incorrect dimensions
  * for example, user ingredient "bug spray" would be filtered out by the vectorizer, but user ingredient "manouri" would put manouri over the min_df=2 threshold
  * tried to fix this by vectorizing with the user input already in, but there was an issue keeping the indexes straight
* since model selection was performed in another notebook, there is no train/test split, using all the data for training

### CODE/FUNCTIONALITY ###

The assignment is split into 7 sections, as follows:

#### Parse Dataset ####

To parse the dataset, the read_json functionality from pandas was utilized. 
The TfidfVectorizer was having a hard time with the tokenized words, so they were de-tokenized

__read_in(filename):__

parameters: name of json file to read in  
returns: a pandas dataframe with 'id', 'cuisine' and 'ingredients' columns  
libraries: pandas

* read_in uses the pandas read_json() function to read the json file into a dataframe

__de_tokenize(dataframe):__

parameters: the recipes dataframe  
returns: the recipes dataframe with the ingredients as one string  
libraries: pandas

* de_tokenize uses ' '.join to join all the individual ingredients together in one string
* the list of these strings is added to the existing dataframe as a column and the dataframe is returned


#### Convert Text to Features ####

The text was vectorized using TfidfVectorizer

__vectorize(dataframe,train):__

parameters: a dataframe to vectorize, and a training flag  
returns: matrix of vectorized ingredient lists, optionally array of cuisines as well  
libraries: pandas, sklearn, numpy

* vectorizer uses TfidfVectorizer from sklearn to vectorize the ingredients column of the recipes dataframe
* if train == True, it also returns a numpy array of the cuisine labels for classificaiton
* if train == False, it only returns the last vector in the Tfidf matrix, since this will be the features for prediction


#### Train Classifier ####


__train_classifier(X,y):__

parameters: a matrix of ingredient vectors, and array of cuisine labels  
returns: a trained classifier  
libraries: sklearn

* train_classifier fits a k-nearest neighbors classifier with 11 neighbors on the ingredient lists and cuisine labels


#### Get Input ####

Because this was done in a notebook, "getting" the input was done manually.

However, in order to effectively predict using the imput, one function was needed.

__parse_user_input(user_input,dataframe):__

parameters: a list of user input ingredients, and the original dataframe  
returns: a dataframe with the new ingredients as the bottom row  
libraries: pandas

* parse_user_input makes a copy of the recipes dataframe
* it then uses ' '.join() to join the list together into a strint
* finally it appends the new ingredients list to the end of the dataframe and returns

#### Prediction ####

The dataframe returned from parse_user_input is vectorized.

Then, the bottom row of that vector is given to the .predict() method of the classifier.

__classify_ingredients(classifier,X):__

parameters: trained classifier, ingredients vector  
returns: predicted cuisine and % of neighbors that have the same label as string  
libraries: sklearn

* the vector is fed into the .predict() method of the classifier to get the cuisine label
* it is then fed into the .predict_proba() method of the classifier to get the % of neighbors with the same cuisine label
* those two pieces of data are returned as a string


#### Select Closest N Recipes ####

__closest_recipes(classifier,X,dataframe,count):__

parameters: trained classifier, ingredients vector, original dataframe and desired number of recipes  
returns: closest recipes and their Euclidean distances 
libraries: sklearn, pandas

* uses the classifiers .kneighbors() function on the X vector to get the indexes of the nearest recipes and their distances
* uses the indexes to find the recipe id number
* returns pairs of id numbers and distances as a string

#### Give Output ####

The main function takes user input and returns the desired output

__main(user_input):__

parameters: user-input ingredient list 
returns: cuisine and closest recipe predictions  
libraries: pandas

* creates and trains classifier from yummly dataset
* predicts cuisine type and closest recipes using above functions
* returns desired output as string

### TESTS ###

As this was done in a notebook, there are no unit tests, but each function was visually tested.

This was done by printing different pieces or attributes of the function return values, as follows:
* __read_in()__: check that the shape of the dataframe is correct, and check the first few entries
* __de_tokenize()__: check that a column has been added to dataframe and check the first few entries
* __vectorize()__: check dimensions of returned matrix and label array; check that single vector has same columns as matrix
* __train_classifier()__: check that the returned classifier is of class KNeighborsClassifier and that n_neighbors == 11
* __parse_user_input()__: check that the input ingredients have been added to the last row of the dataframe
* __classify_ingredients()__: check that a cuisine and probability are returned and that the cuisine is italian
* __closest_recipes()__: check that N closest recipes and their distances are returned

### REFERENCES ###

* scikit-learn documentation, https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html , for kNN info


