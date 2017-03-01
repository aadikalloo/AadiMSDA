library(shiny)
library(plotly)
library(stringr)
library(dplyr)
hw3data = read.csv("https://raw.githubusercontent.com/charleyferrari/CUNY_DATA608/master/lecture3/data/cleaned-cdc-mortality-1999-2010-2.csv", stringsAsFactors = FALSE)
#hw3data = subset(hw3data, hw3data$Year=="2010")
national_averages = hw3data %>% group_by(ICD.Chapter, Year) %>% summarise(national = sum(Population)/sum(Deaths))
# Define server logic required to draw a histogram
shinyServer(function(input, output) {
  
  selectedData <- reactive({
    selected_disorder = input$disorders
    selected_state = input$states
    this_subset = subset(hw3data, hw3data$ICD.Chapter==selected_disorder & hw3data$State == selected_state)
    this_subset = this_subset[order(this_subset$Crude.Rate),]
    this_subset$State = as.factor(this_subset$State)
    this_subset
  })
  
  selectedData_national = reactive({
    selected_disorder = input$disorders
    this_data = subset(national_averages, national_averages$ICD.Chapter==selected_disorder)
  })
  
  output$distPlot <- renderPlot({
    ggplot(selectedData(),aes(x= as.factor(Year), Crude.Rate))+geom_bar(stat ="identity", fill="#000099") + labs(y="Crude Death Rate", x="Year")
  })
  output$nationalPlot = renderPlot ({
    ggplot(selectedData_national(),aes(x= as.factor(Year), national))+geom_bar(stat ="identity", fill="#017004") + labs(y="Crude Death Rate", x="Year")
  })
  
})
