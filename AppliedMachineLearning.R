# R script to find the relation between annotations, tweet legnth and gender of the user
# for removing all variables
rm(list=ls())

# installing and loading all libraries/ packages
install.packages("stringr")
install.packages("tm"); install.packages("SnowballC"); install.packages("wordcloud")
install.packages("corpus")
install.packages("twitteR")
library(twitteR)
library(tm) 
library(SnowballC)
library(wordcloud) 
library('stringr')
library('ggplot2')

# loading processed and raw dataset respectively
twitter_whole_dataset <- read.csv("/Users/nishantsalvi/Downloads/twitter.csv",header = TRUE) 
str(twitter_whole_dataset)
table(twitter_whole_dataset$gender)
View(twitter_whole_dataset)
twitter_whole_dataset <- twitter_whole_dataset[twitter_whole_dataset$gender != "unknown",]
twitter_whole_dataset <- twitter_whole_dataset[!(is.na(twitter_whole_dataset$gender) | twitter_whole_dataset$gender==""), ]
twitter_whole_dataset <- na.omit(twitter_whole_dataset)
table(twitter_whole_dataset$X_golden, twitter_whole_dataset$gender)

# seeing structure and converting raw dataset to char for manipulation
# taking the pre processed dataset
twitter_dataset_char <- data.frame(lapply(twitter_whole_dataset, as.character), stringsAsFactors=FALSE)
str(twitter_dataset_char)

# checking the count of addresses in each
# Hypothesis 1: A particular gender uses more addresses(basically more '@') in a tweet
# max no. of addresses is 3
twitter_dataset_char$at_count <- sapply(twitter_dataset_char$text, function(x) str_count(x, '@'))
maxAt <- max(twitter_dataset_char$at_count)
maxAt
View(twitter_dataset_char$at_count)
twitter_dataset_char$at_countD[twitter_dataset_char$at_count == 0] <- '0'
twitter_dataset_char$at_countD[twitter_dataset_char$at_count == 1] <- '1'
twitter_dataset_char$at_countD[twitter_dataset_char$at_count == 2] <- '2'
twitter_dataset_char$at_countD[twitter_dataset_char$at_count %in% c(3:maxAt)] <- '3+'
#witter_dataset_char$at_countD[twitter_dataset_char$at_count == 3] <- '3'
twitter_dataset_char$at_countD <- factor(twitter_dataset_char$at_countD)
View(twitter_dataset_char)

#twitter_dataset_char$at_countD[twitter_dataset_char$at_count %in% c(3:maxAt)] <- '3+'

# checking hypothesis 2: A particular gender writes tweets of more length
twitter_dataset_char$tweetText_count <- sapply(gregexpr("[[:alpha:]]+", twitter_dataset_char$text), function(x) sum(x > 0))
prop.table(table(twitter_dataset_char$tweet_count))
View(twitter_dataset_char)
table(twitter_dataset_char$gender, twitter_dataset_char$tweetText_count)


#plotting graphs
sentPlt     <- c('#f93822','#fedd00','#27e833')
sentBreaks  <- c('male','female','brand')

ggplot(twitter_dataset_char, aes(x = at_count)) + 
  geom_density(fill = '#99d6ff', alpha=0.4) +
  labs(x = 'Number of @s') +
  theme(text = element_text(size=12))

table(twitter_dataset_char$at_countD, twitter_dataset_char$gender)
table(twitter_dataset_char$gender)
#lapply(twitter_dataset_char$gender, function(x) length(table(x)))
as.character(unique(unlist(twitter_dataset_char$gender)))
twitter_dataset_char$tweetText_count

ggplot(twitter_dataset_char, aes(x = tweetText_count, 
                   fill = gender)) + 
  geom_density(alpha = 0.2) +
  scale_fill_manual(name   = 'Gender',
                    values = sentPlt,
                    breaks = sentBreaks) +
  geom_vline(xintercept = 50, 
             lwd=1, lty = 'dashed') +
  labs(x = 'Tweet Length') +
  theme(text = element_text(size=12))
