library(ggplot2)
library(bigvis)
library(dplyr)
library(plyr)

all_pluto_data = read.csv("/Users/aadi/Google Drive/School/MS Data Analytics/IS608 Data Visualization/Homework2/all_PLUTO_data.csv", stringsAsFactors = FALSE)

year_built_valid = subset(all_pluto_data, all_pluto_data$YearBuilt!=0 & all_pluto_data$YearBuilt>1850)
hist(year_built_valid$YearBuilt, breaks=length(unique(year_built_valid$YearBuilt)))

ggplot(data=year_built_valid, aes(year_built_valid$YearBuilt)) + 
  geom_histogram(binwidth=1, colour="black", fill="white") +
  labs(title="Histogram for Year of Construction") +
  labs(x="Year", y="Count")


num_floors_valid = year_built_valid #subset(all_pluto_data, all_pluto_data$NumFloors!=0)
floors_20 = subset(num_floors_valid, num_floors_valid$NumFloors %in% c(1, 20))
floors_30 = subset(num_floors_valid, num_floors_valid$NumFloors %in% c(21, 30))
floors_40 = subset(num_floors_valid, num_floors_valid$NumFloors %in% c(31, 40))

ggplot(data=floors_20, aes(floors_20$YearBuilt)) + 
  geom_histogram(binwidth=1, fill="navy", alpha=0.6) +
  geom_histogram(data=floors_30, aes(floors_30$YearBuilt), binwidth=1, fill="green", alpha=0.6) +
  geom_histogram(data=floors_40, aes(floors_40$YearBuilt), binwidth=1, fill="red", alpha=0.6) +
  scale_y_continuous(trans="log2", limits=c(1.8,10000), na.value=0) +
  labs(title="Histogram for Year of Construction") +
  labs(x="Year", y="Count") 

num_floors_valid$PPF = num_floors_valid$AssessTot/num_floors_valid$NumFloors
num_floors_valid = num_floors_valid[is.finite(num_floors_valid$PPF),]
f = ddply(num_floors_valid, .(YearBuilt), summarize,  Average=mean(PPF))

ggplot(data=f, aes(x=YearBuilt, y=Average)) + 
  geom_bar(stat="identity") + 
  labs(title="Histogram for Year of Construction vs Cost per floor") +
  labs(x="Year", y="Count")