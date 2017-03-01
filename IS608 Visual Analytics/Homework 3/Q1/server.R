library(shiny)
library(plotly)
hw3data = read.csv("https://raw.githubusercontent.com/charleyferrari/CUNY_DATA608/master/lecture3/data/cleaned-cdc-mortality-1999-2010-2.csv", stringsAsFactors = FALSE)
hw3data = subset(hw3data, hw3data$Year=="2010")

# Define server logic required to draw a histogram
shinyServer(function(input, output) {
  
  selectedData <- reactive({
    f = input$disorders
    this_subset = subset(hw3data, hw3data$ICD.Chapter==f)
    this_subset = this_subset[order(this_subset$Crude.Rate),]
    this_subset$State = as.factor(this_subset$State)
    this_subset
  })

  output$distPlot <- renderPlot({
    ggplot(selectedData(),aes(x= reorder(State,-Crude.Rate), Crude.Rate))+geom_bar(stat ="identity", fill="#000099") + labs(y="Crude Death Rate", x="State")
  })
  
})
