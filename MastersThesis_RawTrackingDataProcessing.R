library(sf)
library(mapview)
library(dplyr)
library(sp)
library(geosphere)

setwd("C:/Users/brian/Downloads/FullGPS") #Set working directory
#Reading in Tracking Device Data
TrackingData = read.csv("TG 2nd Half 060418.csv")
#options(digits = 15) #For sensitivity for lat/long variables
TrackingData = TrackingData[-c(1,2,3,4,8,17,18,19,20)] #remove unwanted columns
summary(TrackingData)

#Normalising elapsed time
y = TrackingData[1,1]
TrackingData$Elapsed.Time = (TrackingData$Elapsed.Time - y)

#Removing time before start of game
TrackingData = TrackingData[which(TrackingData$Elapsed.Time >= 30.77),] #This '13' needs to change depending on each data file
x = TrackingData[1,1]
TrackingData$Elapsed.Time = (TrackingData$Elapsed.Time - x)

#Removing time at end
TrackingData = TrackingData[which(TrackingData$Elapsed.Time <= 2961.46),]

#Aggregating means for every 2.5s
n = 250
DFmeans = aggregate(TrackingData,list(rep(1:(nrow(TrackingData)%/%n+1),each=n,len=nrow(TrackingData))),mean)

#Removing irrelevant variables
DFmeans = DFmeans[-c(5,6)]

#Renaming
colnames(DFmeans)[3:10] = c("Mean_Speed", "Mean_IAI", "Mean_AcclX", "Mean_AcclY", "Mean_AcclZ", "Mean_GyroX", "Mean_GyroY", "Mean_GyroZ") 

#Aggregating minimums for every 2.5s
DFmins = aggregate(TrackingData,list(rep(1:(nrow(TrackingData)%/%n+1),each=n,len=nrow(TrackingData))),min)

#Removing irrelevant variables
DFmins = DFmins[-c(5,6)]

#Renaming
colnames(DFmins)[3:10] = c("Min_Speed", "Min_IAI", "Min_AcclX", "Min_AcclY", "Min_AcclZ", "Min_GyroX", "Min_GyroY", "Min_GyroZ") 

#Aggregating Maximums for every 2.5s
DFmaxs = aggregate(TrackingData,list(rep(1:(nrow(TrackingData)%/%n+1),each=n,len=nrow(TrackingData))),max)

#Removing irrelevant variables
DFmaxs = DFmaxs[-c(5,6)]

#Renaming
colnames(DFmaxs)[3:10] = c("Max_Speed", "Max_IAI", "Max_AcclX", "Max_AcclY", "Max_AcclZ", "Max_GyroX", "Max_GyroY", "Max_GyroZ")

#Dealing with GPS data
GPS = TrackingData[,c("Longitude","Latitude")]

# Convert data frame to sf object
my.sf.point = st_as_sf(x = GPS, 
                       coords = c("Longitude", "Latitude"),
                       crs = "+proj=longlat +datum=WGS84 +ellps=WGS84 +towgs84=0,0,0")

#Simple plot of player movements
plot(my.sf.point)

#Interactive map:
mapview(my.sf.point)

#Aggregating means for every 0.5s
b = 50
GPSmeans = aggregate(GPS,list(rep(1:(nrow(GPS)%/%b+1),each=b,len=nrow(GPS))),mean)

#Deleting rows not needed for COD angle calculations (rows ending in 2,4,7 and 9)

GPSmeans = GPSmeans %>% dplyr::filter(row_number() %% 10 != 2) ## Delete every 10th row starting from 2
GPSmeans = GPSmeans %>% dplyr::filter(row_number() %% 9 != 3) ## Delete every 9th row starting from 3
GPSmeans = GPSmeans %>% dplyr::filter(row_number() %% 8 != 5) ## Delete every 8th row starting from 5
GPSmeans = GPSmeans %>% dplyr::filter(row_number() %% 7 != 6) ## Delete every 7th row starting from 6

xy = GPSmeans[,c(2,3)]

spdf <- SpatialPointsDataFrame(coords = xy, data = GPSmeans,
                               proj4string = CRS("+proj=longlat +datum=WGS84 +ellps=WGS84 +towgs84=0,0,0"))

#Function to allow for specific distances to be vectorised
dist_geo <- function(lat_a, lon_a, lat_b, lon_b) {
  distm(c(lon_a, lat_a), c(lon_b, lat_b), fun = distHaversine) }

#Find distances between every value and next value
result = vector("numeric", length(GPSmeans$Longitude)) # prepare a container
for (i in 1:length(GPSmeans$Longitude)) {
  AandCdists <- mapply(lat_a=GPSmeans$Latitude[i], lon_a=GPSmeans$Longitude[i], lat_b=GPSmeans$Latitude[i+1], lon_b=GPSmeans$Longitude[i+1], FUN = dist_geo)
  result[i] <- AandCdists         # change to assignment
}

#extracting A and C dists
Cdists = result[seq(1, length(result), 3)]
Adists = result[seq(2, length(result), 3)]

#Dealing with B dists
NewGPSmeans = GPSmeans %>% dplyr::filter(row_number() %% 3 != 2) ## Delete every 3rd row starting from 2
resultB <- vector("numeric", length(NewGPSmeans$Longitude)) # prepare a container
for (j in 1:length(NewGPSmeans$Longitude)) {
  Bdist <- mapply(lat_a=NewGPSmeans$Latitude[j],lon_a=NewGPSmeans$Longitude[j],lat_b=NewGPSmeans$Latitude[j+1],lon_b=NewGPSmeans$Longitude[j+1],FUN = dist_geo)
  resultB[j] <- Bdist         # change to assignment
}

#Extracting B dists
Bdists = resultB[seq(1, length(resultB), 2)]

#Merging A,B and C dists
AllDists = cbind(Adists,Bdists, Cdists)

#Creating cosine rule function to find COD angle for each segment
Cosinerule = function(a, b, c){
  (180)-((acos(((a^2)+(c^2)-(b^2))/(2*a*c)))*57.2957795131)
}

#Applying function
CODangle = Cosinerule(a=Adists, b=Bdists, c=Cdists)
summary(CODangle)

#Combining obtained features (means, maxs, mins and COD angle) for each segment
ObtainedFeatures = merge(DFmeans, DFmins, by="Group.1")
ObtainedFeatures = merge(ObtainedFeatures, DFmaxs, by="Group.1")
ObtainedFeatures = cbind(ObtainedFeatures, CODangle)
#Delete mean elapsed time - irrelevant
ObtainedFeatures = ObtainedFeatures[-c(2)]
#Removing missing values
ObtainedFeatures = na.omit(ObtainedFeatures)

#Tidying Up Elapsed time Columns
ObtainedFeatures$Elapsed.Time.y = round(ObtainedFeatures$Elapsed.Time.y, digits = 1)
ObtainedFeatures$Elapsed.Time = round(ObtainedFeatures$Elapsed.Time, digits = 1)
colnames(ObtainedFeatures)[10] = "Start.Elapsed"
colnames(ObtainedFeatures)[19] = "End.Elapsed"