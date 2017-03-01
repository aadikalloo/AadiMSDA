#
# This is the user-interface definition of a Shiny web application. You can
# run the application by clicking 'Run App' above.
#
# Find out more about building applications with Shiny here:
# 
#    http://shiny.rstudio.com/
#

library(shiny)
library(plotly)

hw3data = read.csv("https://raw.githubusercontent.com/charleyferrari/CUNY_DATA608/master/lecture3/data/cleaned-cdc-mortality-1999-2010-2.csv", stringsAsFactors = FALSE)
hw3data = subset(hw3data, hw3data$Year=="2010")
disorders = as.list(unique(hw3data$ICD.Chapter))
names(disorders) = disorders

# Define UI for application that draws a histogram
shinyUI(fluidPage(
  
  # Application title
  titlePanel("CDC Mortality Data"),
  
  # Sidebar with a slider input for number of bins 
  sidebarLayout(
    sidebarPanel(
       selectInput("disorders", "Select a Disorder", choices=disorders)#,
    ),
    
    # Show a plot of the generated distribution
    mainPanel(
       h2("Death Rate Across States for Selected Disorder (Year 2010 only)"),
       plotOutput("distPlot")
    )
  )
))
