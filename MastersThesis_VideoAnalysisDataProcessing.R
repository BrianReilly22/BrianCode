library(readxl)
library(chron)

setwd("C:/Users/brian/Downloads/FullTimestamps") # Setting WD
#Reading In Timestamps data
TimeStamps = read_excel("Gallagher 2nd.xlsx")
Grp1 = TimeStamps[TimeStamps$category=="45-90 DEG",]
Grp2 = TimeStamps[TimeStamps$category=="90-135 DEG",]
Grp3 = TimeStamps[TimeStamps$category=="135-180 DEG",]

#Formatting
TimeStamps = rbind(Grp1, Grp2)
TimeStamps = rbind(TimeStamps, Grp3)
TimeStamps = TimeStamps[-c(4, 5, 6)]

#Converting h:m:s:ms to no. of elapsed seconds 
startMS = substring(TimeStamps$`start time`, 10)
endMS = substring(TimeStamps$`end time`, 10)
startMS = paste0("0.", startMS)
endMS = paste0("0.", endMS)
startMS = as.numeric(startMS)
endMS = as.numeric(endMS)

TimeStamps$`start time` = substr(TimeStamps$`start time`,1,nchar(TimeStamps$`start time`)-3)
TimeStamps$`end time` = substr(TimeStamps$`end time`,1,nchar(TimeStamps$`end time`)-3)

Start.Elapsed = 60 * 60 * 24 * as.numeric(times(TimeStamps$`start time`))
End.Elapsed = 60 * 60 * 24 * as.numeric(times(TimeStamps$`end time`))
Start.Elapsed = Start.Elapsed + startMS
End.Elapsed = End.Elapsed + endMS

#Subtracting time before game begins
Start.Elapsed = Start.Elapsed - 11.46 ## "11.46" needs change for each indivudal file
End.Elapsed = End.Elapsed - 11.46 ## "11.46" needs change for each indivudal file

#Formatting
TimeStamps = cbind(End.Elapsed, TimeStamps)
TimeStamps = cbind(Start.Elapsed, TimeStamps)
TimeStamps = TimeStamps[-c(3,4)]

#Exploring capture durations
diffs = TimeStamps$End.Elapsed - TimeStamps$Start.Elapsed
mean(diffs)

#Getting rid of COD category variable (Binary classifier)
TimeStamps = TimeStamps[-c(3)]

#Attach 'outcome' column of 1s
Outcome = rep(1, length(TimeStamps$Start.Elapsed))
TimeStamps = cbind(TimeStamps, Outcome)