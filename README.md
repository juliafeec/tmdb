# TMDb Movie Genres Classification



## Goal
We want to predict movie genres based on general movie information obtained from TMDb website.  

The point of doing this is that the movie genres on the movie review websites such as TMDb and IMDb are open to anyone to edit. This causes some popular movies to have more genres than it is supposed to have, while some unpopular movies only have one or even no genre. This will be very inconvenient for users who want to find a movie by genre. If the movie genres can be correctly predicted using other information we have, we can then give appropriate suggestions to users who want to add genres to a movie, or better deal with movies with no genres.  
 

## Data Description
**Base Dataset**:  
<https://www.kaggle.com/tmdb/tmdb-movie-metadata#tmdb_5000_movies.csv>

This dataset is provided by TMDb Movie Database. It contains general information of around 5000 movies released from early 20th century to 2017. The information inludes movie title, cast, overview, keywords, genres, budget, revenue, release date, language, runtime, vote, etc. Those information come in two separate csv files, with a unique column `movie_id`.


**Subtitles**:  
<https://github.com/Diaoul/subliminal> (API to download subtitles)  

We also include the subtitles because it could be a powerful feature describing the genres of a movie. Since we will be using word2vec in our model, we only collect subtitles that are in English using the python API `subliminal`.  


## Data Preprocessing

### TMDb movie data
Since this part of the data are raw data from TMDb, there are both missing values and outliers.  

+ For text-format missing values, especially for `genres` and `overview`, we choose to drop those observations because we cannot estimate those values from other observations.  

+ For numerical features - `budget` and `revenue` - we observed some outliers which obviously do not make sense, for example, \$10 movie budget and \$100 movie revenue. For these values, we used a Random Forest Regressor to estimate and replace these value based on `overview`,  `popularity` and `release_date`.  

### Subtitles
The original format for subtitles are `.srt`, including both subtitles and the timestamp of each sentence. We remove the timestamps but we keep the same sentence and paragraph split because we are going to do `Doc2Vec` on subtitles.


## Feature Engineering

+ **Overview**:  
	For overviews, we tried both `Tfdif` and `Doc2Vec` to vectorize the text. It turns out that `Tfidf` works better maybe because the length of overview text is not enough to train the `Doc2Vec` vectors well.  
	
+ **Subtitles**:  
	In addition to `Doc2Vec`, we also did `Tfdif` on subtitles because we want to know which words have the largest influence on deciding the movie genres.  
	
+ **Cast names**:  
	For cast names, we remove the space between the first name and last name to make each name a unique work and use `CountVectorizer` to vectorize them.  
	
+ **Keywords, production companies, production countries**:  
	We train a `MultiLabelBinarizer` and do one-hot encoding for these three features.  

## Modeling



## Evaluation

We choose precision, recall and hamming loss as our evaluation metrics.  

+ **Precision, Recall, F-score**:   
	We care about the precision because we don't want to confuse the users and make sure that the movies they find using genres would meet their expectations as much as possible.   
	We care about the recall per class because in practice we want to add as many "correct" movie genres as possible to better label the movies.   

+ **Hamming loss**:  
	For multi-label problems, hamming loss is a good metrics to evaluate the performance of the model.   
	
+ **Accuracy**:
