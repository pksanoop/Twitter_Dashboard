#!/usr/bin/env python
# coding: utf-8

# # Big Data Project: Twitter Sentiment Analysis 
# 
# ### Sanoop Kammampata

# In[ ]:


get_ipython().system('pip install textblob')


# ### 1. Load Required Libraries

# In[ ]:


# libraries for loading, cleaning, and processing the data
import pandas as pd
import numpy as np 
from textblob import TextBlob
from datetime import datetime
import pytz
import sklearn
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, TimestampType, FloatType
import pyspark.sql.functions as F
from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import HashingTF, IDF
from pyspark.ml.feature import StringIndexer

# # libraries for ML part
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import NGram, VectorAssembler, StopWordsRemover, HashingTF, IDF, Tokenizer, StringIndexer, ChiSqSelector, VectorAssembler, CountVectorizer
from pyspark.ml import Pipeline


# ### 2. Data
# 
# The data (BlackFriday) used in this project was taken from amazon S3 bucket.

# #### 2.1 Load Data From AWS S3

# In[ ]:


def mount_s3_bucket(access_key, secret_key, bucket_name, mount_folder):
  ACCESS_KEY_ID = access_key
  SECRET_ACCESS_KEY = secret_key
  ENCODED_SECRET_KEY = SECRET_ACCESS_KEY.replace("/", "%2F")

  print ("Mounting", bucket_name)

  try:
    # Unmount the data in case it was already mounted.
    dbutils.fs.unmount("/mnt/%s" % mount_folder)
    
  except:
    # If it fails to unmount it most likely wasn't mounted in the first place
    print ("Directory not unmounted: ", mount_folder)
    
  finally:
    # Lastly, mount our bucket.
    dbutils.fs.mount("s3a://%s:%s@%s" % (ACCESS_KEY_ID, ENCODED_SECRET_KEY, bucket_name), "/mnt/%s" % mount_folder)
    #dbutils.fs.mount("s3a://"+ ACCESS_KEY_ID + ":" + ENCODED_SECRET_KEY + "@" + bucket_name, mount_folder)
    print ("The bucket", bucket_name, "was mounted to", mount_folder, "\n")
    
    
# Set AWS programmatic access credentials
ACCESS_KEY = "AKIDVRXD7DZLSR9CT7VS"
SECRET_ACCESS_KEY = "+02sqCkvLQA4JDM6TLPyf0J6e91uJOEz+vsrcOLQ"


# In[ ]:


mount_s3_bucket(ACCESS_KEY, SECRET_ACCESS_KEY, 'weclouddata/twitter/BlackFriday', 'BlackFriday')


# In[ ]:


get_ipython().run_line_magic('fs', 'ls /mnt/BlackFriday')


# In[ ]:


path = 'mnt/BlackFriday/*/*/*/*/*'


# In[ ]:


# Create Spark session
spark = SparkSession \
        .builder \
        .appName('big_data_project') \
        .getOrCreate()
print('Session created')
sc = spark.sparkContext


# In[ ]:


# Create Schema
Schema = StructType([
    StructField('id', StringType(), True),
    StructField('name', StringType(), True),
    StructField('screen_name', StringType(), True),
    StructField('tweet', StringType(), True),
    StructField('followers_count', IntegerType(), True),
    StructField('location', StringType(), True),
    StructField('geo', StringType(), True),
    StructField('created_at', StringType(), True)
])
 


# In[ ]:


# Read dataframe
df = (spark
     .read
     .option('header', 'false')
     .option('delimiter', '\t')
     .schema(Schema)
     .csv(path))


# In[ ]:


# Cache the dataframe for faster iteration and get the count
df.cache()
df.count()


# In[ ]:


display(df.take(50))


# In[ ]:


# Mount the data to my bucket
mount_s3_bucket(ACCESS_KEY, SECRET_ACCESS_KEY, 'bc16/BlackFriday/', 'my_bucket')


# In[ ]:


# Save as csv file in my bucket
(df
 .write
 .option('header', 'false')
 .option('delimiter', '\t')
 .csv('/mnt/my_bucket/BlackFriday.csv'))


# #### 2.2 Text Cleaning and Preprocessing the Data
# 
# 1. Remove URLs
# 2. Remove special characters
# 3. Substituting multiple spaces with single space
# 4. Lowercase all text
# 5. Trim the leading/trailing whitespaces

# In[ ]:


df_clean = df.withColumn('tweet', F.regexp_replace('tweet', r"http\S+", "")) \
                    .withColumn('tweet', F.regexp_replace('tweet', r"[^a-zA-z]", " ")) \
                    .withColumn('tweet', F.regexp_replace('tweet', r"\s+", " ")) \
                    .withColumn('tweet', F.lower('tweet')) \
                    .withColumn('tweet', F.trim('tweet')) 
display(df_clean)  


# In[ ]:


# Check for null values
df_clean.select([F.count(F.when(F.isnan(c),c)).alias(c) for c in df_clean.columns]).toPandas().head()


# In[ ]:


# Drop rows with null values and check the count
#df_drop = df_clean.na.drop(how='any')
df_drop.count()


# In[ ]:


# Drop duplicates and check the count 

df_duplicates = df_drop.dropDuplicates()
df_duplicates.count()


# In[ ]:


# Use TextBlob to assign the labels

def get_sentiment(tweet):
    blob = TextBlob(tweet)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        return 'positive'
    else:
        return 'negative'


# In[ ]:


# Add sentiment column
get_sentiment_udf = udf(get_sentiment, StringType())
df_label = df_duplicates.withColumn('Sentiment', get_sentiment_udf('tweet'))
display(df_label.take(50))


# In[ ]:


# Save as csv file in my bucket
(df_label
 .write
 .option('header', 'true')
 .option('delimiter', '\t')
 .mode('overwrite')
 .csv('/mnt/my_bucket/BlackFriday_cleaned.csv'))


# #### 2.3 Feature Transformer: Tokenizer

# In[ ]:


tokenizer = Tokenizer(inputCol='tweet', outputCol='tokens')
tweets_tokenized = tokenizer.transform(df_label)

display(tweets_tokenized.take(50))


# #### 2.4 Feature Transformer: Stopword Removal

# In[ ]:


# Remove stopwords from the review(list of words)    

stopword_remover = StopWordsRemover(inputCol='tokens', outputCol='filtered')
tweets_stopword = stopword_remover.transform(tweets_tokenized)

display(tweets_stopword.take(50))


# #### 2.5 Feature Transformer: CountVectorizer (TF - Term Frequency)

# In[ ]:


cv = CountVectorizer(vocabSize=2**16, inputCol='filtered', outputCol='cv')
cv_model = cv.fit(tweets_stopword)
tweets_cv = cv_model.transform(tweets_stopword)

display(tweets_cv.take(50)) 


# #### 2.6 Feature Transformer: TF-IDF Vectorization

# In[ ]:


idf = IDF(inputCol='cv', outputCol='features', minDocFreq=5) #minDocFreq: remove sparse terms
idf_model = idf.fit(tweets_cv)
tweets_idf = idf_model.transform(tweets_cv)

display(tweets_idf.take(50))


# #### 2.7 Label Encoder

# In[ ]:


# Use label encoder since sentiment is string.

label_encoder = StringIndexer(inputCol = 'Sentiment', outputCol = 'label') 
le_model = label_encoder.fit(tweets_idf)
tweets_label = le_model.transform(tweets_idf)

display(tweets_label.take(50))


# In[ ]:


# Cache the dataframe for faster iteration and get the count
tweets_label.cache()
tweets_label.count()


# In[ ]:


tweets_label.select('sentiment', 'label').show(10)


# In[ ]:


tweets_label.groupBy('label').count().show()


# In[ ]:


# Save as parquet file in my bucket
(tweets_label
 .write
 .mode('overwrite')
 .parquet('/mnt/my_bucket/BlackFriday_label.parquet'))


# ### 3. Develop Machine Learning Model

# #### 3.1 Model Training: Logistic Regression Classifier

# In[ ]:


# Use 90% cases for training, 10% cases for testing
train, test = tweets_label.randomSplit([0.9, 0.1], seed=20200819)

lr = LogisticRegression(maxIter=100)

lr_model = lr.fit(tweets_label)

predictions = lr_model.transform(tweets_label)

display(predictions)


# #### 3.2 Model evaluation

# In[ ]:


evaluator = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction')
roc_auc = evaluator.evaluate(predictions)
accuracy = predictions.filter(predictions.label == predictions.prediction).count() / float(predictions.count())

print("Accuracy Score: {0:.4f}".format(accuracy))
print("ROC-AUC: {0:.4f}".format(roc_auc))


# In[ ]:


# Save predictions as parquet file in my bucket
(predictions
 .write
 .mode('overwrite')
 .parquet('/mnt/my_bucket/BlackFriday_predictions.parquet'))


# In[ ]:


display(predictions) 


# In[ ]:


# Check the schema of predictions
predictions.printSchema()


# #### 3.3 Putting a Pipeline Together

# In[ ]:


# Use 90% cases for training, 10% cases for testing
train, test = df_label.randomSplit([0.9, 0.1], seed=20200819)

# Create transformers for the ML pipeline 
tokenizer = Tokenizer(inputCol='tweet', outputCol='tokens')
stopword_remover = StopWordsRemover(inputCol='tokens', outputCol='filtered')
cv = CountVectorizer(vocabSize=2**16, inputCol='filtered', outputCol='cv')
idf = IDF(inputCol='cv', outputCol='1gram_idf', minDocFreq=5) #minDocFreq: remove sparse terms
assembler = VectorAssembler(inputCols=['1gram_idf'], outputCol='features')
label_encoder= StringIndexer(inputCol = 'Sentiment', outputCol = 'label')
lr = LogisticRegression(maxIter=100)
pipeline = Pipeline(stages=[tokenizer, stopword_remover, cv, idf, assembler, label_encoder, lr])

pipeline_model = pipeline.fit(train)
predictions = pipeline_model.transform(test)

evaluator = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction')
accuracy = predictions.filter(predictions.label == predictions.prediction).count() / float(test.count())
roc_auc = evaluator.evaluate(predictions)

print("Accuracy Score: {0:.4f}".format(accuracy))
print("ROC-AUC: {0:.4f}".format(roc_auc))


# In[ ]:


# Save Pipeline as parquet file in my bucket
(predictions
 .write
 .mode('overwrite')
 .parquet('/mnt/my_bucket/BlackFriday_pipe.parquet'))


# #### 3.4 Ngram Features

# In[1]:


# Use 90% cases for training, 10% cases for testing
train, test = df_label.randomSplit([0.9, 0.1], seed=20200819)

# label
label_encoder= StringIndexer(inputCol = 'Sentiment', outputCol = 'label')

# Create transformers for the ML pipeline
tokenizer = Tokenizer(inputCol='tweet', outputCol='tokens')
stopword_remover = StopWordsRemover(inputCol='tokens', outputCol='filtered')
cv = CountVectorizer(vocabSize=2**16, inputCol='filtered', outputCol='cv')
idf = IDF(inputCol='cv', outputCol='1gram_idf', minDocFreq=5) #minDocFreq: remove sparse terms
ngram = NGram(n=2, inputCol='filtered', outputCol='2grami')
ngram_hashingtf = HashingTF(inputCol='2gram', outputCol='2gram_tf', numFeatures=20000)
ngram_idf = IDF(inputCol='2gram_tf', outputCol='2gram_idf', minDocFreq=5) 

# Assemble all text features
assembler = VectorAssembler(inputCols=['1gram_idf', '2gram_tf'], outputCol='rawFeatures')

# Chi-square variable selection
selector = ChiSqSelector(numTopFeatures=2**14,featuresCol='rawFeatures', outputCol='features')

# Regression model estimator
lr = LogisticRegression(maxIter=100)

# Build the pipeline
pipeline = Pipeline(stages=[label_encoder, tokenizer, stopword_remover, cv, idf, ngram, ngram_hashingtf, ngram_idf, assembler, selector, lr])

# Pipeline model fitting
pipeline_model = pipeline.fit(train)
predictions = pipeline_model.transform(test)

evaluator = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction')
accuracy = predictions.filter(predictions.label == predictions.prediction).count() / float(test.count())
roc_auc = evaluator.evaluate(predictions)

print("Accuracy Score: {0:.4f}".format(accuracy))
print("ROC-AUC: {0:.4f}".format(roc_auc))  

