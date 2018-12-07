# TMDb Movie Genres Classification

*Group member: Julia Tavares, Wei Wei, Xiao Han, Jialiang Shi, Xueyin Wang*  

## Table of Contents
* File Description  
* Goal  
* Data Description  
* Data Preprocessing  
* Feature Engineering  
* Modeling  
* Evaluation  
	
## File Description  
* Data Discover:  
	- [TMDB.ipynb](https://github.com/juliafeec/tmdb/blob/master/data_dicover/TMDB.ipynb)  
	- [TMDB Discover.ipynb](https://github.com/juliafeec/tmdb/blob/master/data_dicover/TMDB%20Discover.ipynb)  
	- [TMDB Discover - Part 2 (Join Data).ipynb](https://github.com/juliafeec/tmdb/blob/master/data_dicover/TMDB%20Discover%20-%20Part%202%20(Join%20Data).ipynb)  
* Feature preocessing:  
	- [Download subtitles.ipynb](https://github.com/juliafeec/tmdb/blob/master/feature_preprocessing/Download%20subtitles.ipynb)
	- [Process Subtitles.ipynb](https://github.com/juliafeec/tmdb/blob/master/feature_preprocessing/Process%20Subtitles.ipynb)
	- [Doc2Vec.ipynb](https://github.com/juliafeec/tmdb/blob/master/feature_preprocessing/Doc2Vec.ipynb)  
	- [tokenizer.ipynb](https://github.com/juliafeec/tmdb/blob/master/feature_preprocessing/tokenizer.ipynb)  
* Complete Main Work:  
	- [main_work_2tfidf.ipynb](https://github.com/juliafeec/tmdb/blob/master/main_work_2tfidf.ipynb) Converting both overview and subtitles with tfidf. 
	- [main_work_tfidf+Doc2Vec.ipynb](https://github.com/juliafeec/tmdb/blob/master/main_work_tfidf%2BDoc2Vec.ipynb) Converting overview with tfidf and subtitles with doc2vec . 
* Presentation:   
	- [prensentation_movie.ipynb](https://github.com/juliafeec/tmdb/blob/master/presentation/prensentation_movie.ipynb)  
	- [images](https://github.com/juliafeec/tmdb/tree/master/images)  
	- [WordCloud.ipynb](https://github.com/juliafeec/tmdb/blob/master/presentation/WordCloud.ipynb)  

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

**Not Using Doc2Vec for subtitles:**

| Metrics                  | Multinomial Naive Bayes | Random Forest |
| :----------------------: | :---------------------: | :-----------: |
| Weighted Precision       | 0.5333                  | **0.7551**    |  
| Weighted Recall          | **0.6296**              | 0.3151        |
| Weighted F1 Score        | **0.5743**              | 0.4101        |
| Hamming Loss             | 0.1240                  | **0.1045**    |
| Subset Accuracy          | 0.1327                  | 0.1327        |

**Using Doc2Vec for subtitles:**

| Metrics                  | Multinomial Naive Bayes | Random Forest |
| :----------------------: | :---------------------: | :-----------: |
| Weighted Precision       | 0.4877                  | **0.7610**    |
| Weighted Recall          | **0.6064**              | 0.4374        |
| Weighted F1 Score        | **0.5384**              | 0.5341        |
| Hamming Loss             | 0.1473                  | **0.0936**    |
| Subset Accuracy          | 0.1091                  | **0.1754**    |


## Metrics  

We choose precision, recall and hamming loss as our evaluation metrics.  

+ **Precision, Recall, F-score**:   
	+ We care about the precision because we don't want to confuse the users and make sure that the movies they find using genres would meet their expectations as well as possible.  
  
	+ We care about the recall because in practice we want to add as many "correct" movie genres as possible to better label the movies.   

+ **Hamming loss**:  
	Hamming loss refers to the fraction of the wrong labels to the total number of labels per prediction. So it is a good metric to evaluate model performance in multi-label classification problems.   
	
+ **Accuracy**:   
	Accuracy is not a good metric for multi-label classification problem, since the `accuracy_rate()` in `sklearn` will measure the subset accuracy in this case, and only an exact match will be counted as a 1.
