# NYT Restaurant Reviews
Mod 4 NLP partner project <br/>
By: Sam Jackson and Maks Pazuniak

## Project Goals
In this project, we explore the capabilities of NLP to analyze New York Times restaurant reviews and attempt to predict the star rating for the subject restaurant of a given article

## Data
We scraped the [restaurant review search site](https://www.nytimes.com/reviews/dining) for review URLs, restaurant names, star rating, neighborhood, etc. <br/>
We then found & scraped [this site](https://www.nytimes.com/column/restaurant-review) after realizing the list was more comprehensive, including reviews for restaurants that had been reviewed multiple times. <br/>
We then scraped each review URL for restaurant review, headline, and any features that we hadn't retrieved in our initial scrape.

**Original Dataset:** 986 Reviews <br/>
**Functional Dataset:** 713 Reviews <br/>
**Reviewed By: Pete Wells, Frank Bruni, Sam Sifton:** 615 Reviews

**Features:**
- Reviewer: Author of article
- Vocabulary: # of unique tokens per article (after removing stopwords and lemmatizing)
- Area: Area of NYC where restaurant is located
- blob_avg_pol: Average polarity of sentences in the article, based on TextBlob
- blob_std_pol: Standard Deviation of polarity of sentences in the article, based on TextBlob
- headline_sent: Sentiment of Headline, based on TextBlob
- TF-IDF of the most polar words (>abs(+/-3))
- Reviewer x Average review polarity
- Reviewer x Vocabulary

**Target:**
- Star Rating: Number of stars given to a restaurant by a reviewer. <br/>
*Note: We assumed any review labeled 'Poor', 'Satisfactory', or 'Good' were given 0 stars.  Due to scarcity, we grouped reviews with 3 or 4 stars into one category (labeled 3 stars). These assumptions may have contributed to the difficulties we faced in classifying review rating.*

## Baseline Models
We used TF-IDF and a Random Forest Classifier to come up with a baseline model.  Our baseline had an test-accuracy of .4805.

## EDA
### Reviewers
We selected to look at only the top three reviewers, based on the reviews we scraped.  These three reviewers produced over 80% of the reviews in our "functional" dataset.  

**Pete Wells:**
- 304 Reviews
- 09/2009 - Present
- 2 Star Rating %age: 48%
- Average Star Rating: 1.71 Stars

**Frank Bruni:**
- 218 Reviews
- 06/2004 - 08/2009
- 2 Star Rating %age: 33%
- Average Star Rating: 1.51 Stars
 
**Sam Sifton:**
- 93 Reviews
- 10/2009 - 10/2011
- 2 Star Rating %age: 35%
- Average Star Rating: 1.50 Stars 

![Reviewers](/png_files/RatingsByReviewer.png)

### Sentiment Analysis 
We tested several methods for assessing the polarity of the articles. VADER, since it was trained on & built for Twitter-style text (short document length, caps and punctuation to boost sentiment, emojis, etc), we found that it didn't detect the sentiment of a NYT article as well as some other lexicons, such as TextBlob and Afinn.  We tested models using both TextBlob and Afinn sentiment analysis.  Below, you can see that Afinn is slightly more consistent than TextBlob with expectations - median sentiment increases with star rating.

![Polarity](/png_files/polarity_scoring.png)

We then ran TF-IDF on the corpus of reviews, ran sentiment analysis on each n-gram, and used TF-IDF scores from the most polar (>|+/-3|) n-grams as features in our model. Below is a list of the most polar n-grams:

![PolarWords](/png_files/PolarWords.png)

## Best Model
Our best model predicted the star rating of our test-set of reviews with an accuracy of .536. <br/>
The best model was a Random Forest Classifier and TextBlob sentiment analysis

![best model](/png_files/best_model.png)

Our biggest issue while modeling was the class imbalance among star ratings.  Attmpting to correct the class imbalance with over-sampling led to even worse results.  Under-sampling would have led to too few data points.  

![imbalance](/png_files/ClassImbalance.png)

## Going Forward
We have so much more work we'd like to do on this project! The power of NLP is immense and we would have loved more time to dive deeper.  
- Cosine Similarity to find semantic similarity between articles (and maybe, therefore, star-rating)
- Topic Modeling: do certain cuisines, dishes, ambiance, decor, location, etc impact the star-rating
- More work/implementation of most important words as features:

![important words](/png_files/ImportantWords.png)
