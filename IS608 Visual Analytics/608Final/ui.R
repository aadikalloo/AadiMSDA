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

shinyUI(mainPanel(
  fluidPage(plotlyOutput("plot"),
            verbatimTextOutput("click")),
  fluidPage(column(
    width = 12,
    conditionalPanel(
      condition = "!$('html').hasClass('shiny-busy')",
      h3(textOutput("artist")),
      #to style to d3 output pull in css
      tags$head(
        tags$link(rel = "stylesheet", type = "text/css", href = "style.css")
      ),
      #load D3JS library
      tags$script(src = "https://d3js.org/d3.v3.min.js"),
      #load javascript
      tags$script(src = "d3script.js"),
      #create div referring to div in the d3script
      tags$div(id = "state_name"),
      tags$div(id = "d3_graph1")
    )
  ))
))
