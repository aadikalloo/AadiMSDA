library(shiny)
library(plotly)
library(jsonlite)

trim <- function (x)
  gsub("^\\s+|\\s+$", "", x)

abb2state <- function(name,
                      convert = F,
                      strict = F) {
  data(state)
  # state data doesn't include DC
  state = list()
  state[['name']] = c(state.name, "District Of Columbia")
  state[['abb']] = c(state.abb, "DC")
  
  if (convert)
    state[c(1, 2)] = state[c(2, 1)]
  
  single.a2s <- function(s) {
    if (strict) {
      is.in = tolower(state[['abb']]) %in% tolower(s)
      ifelse(any(is.in), state[['name']][is.in], NA)
    } else{
      # To check if input is in state full name or abb
      is.in = rapply(state, function(x)
        tolower(x) %in% tolower(s), how = "list")
      state[['name']][is.in[[ifelse(any(is.in[['name']]), 'name', 'abb')]]]
    }
  }
  sapply(name, single.a2s)
}


#https://gist.github.com/ligyxy/acc1410041fe2938a2f5


# h1b = read.csv(
#   "/Users/aadi/Google Drive/School/MS Data Analytics/IS608 Data Visualization/Final/h1b_kaggle.csv",
#   stringsAsFactors = FALSE
# )
load("h1b.RData")

h1b$state = sapply(strsplit(as.character(h1b$WORKSITE), "\\,"), "[[", 2)
h1b$state = trim(h1b$state)
ignore_states = c("NA", "PUERTO RICO", "DISTRICT OF COLUMBIA") #these aren't on the map

#h1b_recent = h1b[h1b$YEAR=="2016",]
state_wage = aggregate(PREVAILING_WAGE ~ state, data = h1b, mean)
state_wage = state_wage[!(state_wage$state %in% ignore_states), ]

#df <- read.csv("https://raw.githubusercontent.com/plotly/datasets/master/2011_us_ag_exports.csv")
state_wage$hover <-
  with(state_wage, paste(state, '<br>', "Prevailing Wage: $", floor(PREVAILING_WAGE)))
state_wage$code = abb2state(state_wage$state, convert = T)

# Define server logic required to draw a histogram
shinyServer(function(input, output, session) {
  selectedData <- reactive({
    
  })
  
  output$plot <- renderPlotly({
    #ggplot(selectedData(),aes(x= reorder(State,-Crude.Rate), Crude.Rate))+geom_bar(stat ="identity", fill="#000099") + labs(y="Crude Death Rate", x="State")
    l <- list(color = toRGB("white"), width = 4)
    # specify some map projection/options
    g <-
      list(
        scope = 'usa',
        projection = list(type = 'albers usa'),
        showlakes = FALSE,
        lakecolor = toRGB('white')
      )
    
    plot_geo(state_wage, locationmode = 'USA-states') %>%
      add_trace(
        z = ~ PREVAILING_WAGE,
        text = ~ hover,
        locations = ~ code,
        color = ~ PREVAILING_WAGE,
        colors = 'Blues'
      ) %>%
      colorbar(title = "$ USD") %>% layout(title = 'Prevailing Wage By State, 2016<br>(Hover for breakdown)', geo = g)
  })
  
  output$click <- renderPrint({
    d <- event_data("plotly_click")
    selected_state = "Florida"
    if (is.null(d))
      print("Click on a state to view data")
    else
      #print(state_wage$state[as.numeric(d[2][1]) + 1]);
      selected_state = state_wage$state[as.numeric(d[2][1]) + 1]
      data_subset = h1b[h1b$state == selected_state,]
      ds = aggregate(PREVAILING_WAGE ~ YEAR, data=data_subset, mean)
      ds$state = selected_state
      dataObj = jsonlite::toJSON(ds)
      session$sendCustomMessage(type="jsondata", dataObj)
  })
  
})
