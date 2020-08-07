library("rjson")
library(purrr)

#Read in test data
test = read.csv('test.csv', header = TRUE)

#EMPTY DATAFRAME
df <- data.frame(PetID=character(),
                 SentimentScore=double(),
                 SentimentMagnitude=double()
)

df$PetID = as.character(df$PetID)

#FILES = NAMES OF JSONS
path <- "C:/Users/Brian/Documents/LJMU/SEMESTER 2/DataMining/Coursework/test_sentiment"
files <- dir(path, pattern = "*.json")

#ADD FIRST ROW of df MANUALLY
name = files[2]
result = fromJSON(file = name)
sc = result$documentSentiment$score
mag = result$documentSentiment$magnitude
df[1, ] <- c(substr(name,1,6),sc,mag)

#other rows with for loop
for (i in 3:5038) {
  name = files[i]
  result = fromJSON(file = name)
  sc = result$documentSentiment$score
  mag = result$documentSentiment$magnitude
  df = rbind(df, list(substr(name,1,6),sc,mag))
  rm(result)
  rm(sc)
  rm(mag)
}

#MERGE TWO DATAFRAMES
newdataframe = merge(test, df, by=c("PetID"), all = TRUE)
newdataframe$SentimentScore = as.numeric(newdataframe$SentimentScore)

#EXPORT CSV
write.csv(x=newdataframe, file = "C:/Users/Brian/Documents/LJMU/SEMESTER 2/Data
Mining/Coursework/test_sentiment/NewTest.csv", row.names = FALSE)