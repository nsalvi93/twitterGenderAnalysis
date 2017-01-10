library(ggplot2) # Data visualization
install.packages("readr")
library(readr) # CSV file I/O, e.g. the read_csv function
library(corrplot)

twitter_whole_dataset <- read.csv("twitter.csv",header = TRUE) 
twitter_whole_dataset <- na.omit(twitter_whole_dataset)
twitter_whole_dataset <- unique(twitter_whole_dataset)

correlation <- cor(twitter_whole_dataset[sapply(twitter_whole_dataset, is.numeric)])
corrplot(correlation, method="color")
