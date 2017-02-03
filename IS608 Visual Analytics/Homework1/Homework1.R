#DATA 608 Knowledge and Visual Analytics Homework 1
#Aadi Kalloo
#Due January 5 2017

library(plyr)
library(ggplot2)
library(dplyr)

#Question 1
data = read.csv("https://raw.githubusercontent.com/charleyferrari/CUNY_DATA608/master/lecture1/Data/inc5000_data.csv", stringsAsFactors = FALSE)
state_count = plyr::count(data, vars="State")
state_count = state_count[order(-state_count$freq),]
row.names(state_count) = NULL

ggplot(data=state_count, aes(x=reorder(State, freq), y=freq)) + 
  coord_flip() +
  geom_bar(stat="identity", fill="steelblue")+
  geom_text(aes(label=freq), hjust=-0.3, color="black", size=3.5) +
  labs(title="Distribution of Companies by State", x="State", y="Frequency")


#Question 2
state_3 = state_count$State[3]
state_3_df = data[data$State==state_3,]
state_3_df = state_3_df[complete.cases(state_3_df),]
state_3_df_group_summary = ddply(state_3_df, .(Industry), summarize,  Average=mean(Employees))
ylim1 = c(0, boxplot.stats(state_3_df_group_summary$Average)$out[1])

ggplot(data=state_3_df, aes(x=reorder(Industry, Employees), y=Employees)) + 
  geom_boxplot(outlier.size = NA) + 
  coord_flip(ylim=ylim1) + 
  labs(title=paste0("Distribution of Employees by Industry for ", state_3, " State"))


#Question 3
c_data = data[complete.cases(data),]
c_data_gs = ddply(c_data, .(Industry), summarize,  Average_Income=mean(Revenue/Employees))
c_data_gs1 = ddply(c_data, .(Industry), summarize,  Average_Income=sum(Revenue)/sum(Employees))
ylim2 = c(0, boxplot.stats(c_data_gs$Average_Income)$out[1])


ggplot(data=c_data_gs1, aes(x=reorder(Industry, Average_Income), y=round(Average_Income))) + 
  coord_flip(ylim=ylim2*2) +
  geom_bar(stat="identity", fill="steelblue")+
  geom_text(aes(label=round(Average_Income)), hjust=-0.3, color="black", size=3.5) +
  labs(title="Revenue per Employee by Industry", x="Industry", y="Revenue per Employee")

ggplot(data=c_data, aes(x=reorder(Industry, Revenue/Employees), y=Revenue/Employees)) + 
  geom_boxplot(outlier.size = NA) + 
  coord_flip(ylim=ylim2*2.5) + 
  labs(title=paste0("Distribution of Revenue per Employee by Industry"), x="Industry", y="Revenue per Employee")
