#Syncing timelines of tracking data and timestamps
#Taking times from tracking device data
time = ObtainedFeatures[c("Start.Elapsed", "End.Elapsed")]

#Attaching 'outcome column of 0s
zeros = rep(0, length(time$Start.Elapsed))
time = cbind(time, zeros)
colnames(time)[3] = "Outcome"

#merging timestamps and tracking device timelines
trial = rbind(TimeStamps, time)

#Ordering by time
trial = trial %>% arrange(Start.Elapsed)

#Combining Timelines
trial = trial %>% 
  mutate(Outcome = +(Start.Elapsed < lag(End.Elapsed, default = first(Start.Elapsed))| 
                       as.logical(lead(Outcome, default = 0)))) %>%
  filter((round(End.Elapsed - Start.Elapsed, 1) == 2.5) | (round(End.Elapsed - Start.Elapsed, 1) == 2.4))

newdf = merge(time, trial, by=c("Start.Elapsed", "End.Elapsed"))
newdf = newdf[ -c(3)]

#merging with obtained features and tidying
TrainingDF = merge(ObtainedFeatures, newdf, by = c("Start.Elapsed", "End.Elapsed"))
TrainingDF = TrainingDF %>% arrange(Start.Elapsed)

#Transforming outcome variable to factor instead of numeric continuous
TrainingDF$Outcome.y = as.factor(TrainingDF$Outcome.y)

#Removing uneccessary 'Group' variable
TrainingDF = TrainingDF[-c(3)] 

#Adding PLayerID column
ID = as.factor(rep(19,length(TrainingDF$Start.Elapsed)))
TrainingDF = cbind(ID, TrainingDF) #Final training file

#Exporting
write.csv(x=TrainingDF, file = "C:/Users/brian/Documents/TrainingData/TG2.csv", row.names = FALSE)