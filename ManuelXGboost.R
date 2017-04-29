install.packages("xgboost")
install.packages("caret")
install.packages("RCurl") #sudo apt install libcurl4-openssl-dev
install.packages("Metrics")
library(caret) # for dummyVars
library(RCurl) # download https data
library(Metrics) # calculate errors
library(xgboost) # model

urlfile <- 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
x <- getURL(urlfile, ssl.verifypeer = FALSE)
adults <- read.csv(textConnection(x), header=F)
edit(adults)
names(adults)=c('age','workclass','fnlwgt','education','educationNum',
                'maritalStatus','occupation','relationship','race',
                'sex','capitalGain','capitalLoss','hoursWeek',
                'nativeCountry','income')
adults$income <- ifelse(adults$income==' <=50K',0,1)

# clean up data
adults$income <- ifelse(adults$income==' <=50K',0,1)

# binarize all factors
library(caret)
dmy <- dummyVars(" ~ .", data = adults)
adultsTrsf <- data.frame(predict(dmy, newdata = adults))
#edit(adultsTrsf)

#what we're trying to predict adults that make more than 50k
outcomeName <- c('income')
# list of features
predictors <- names(adultsTrsf)[!names(adultsTrsf) %in% outcomeName]
predictors

# play around with settings of xgboost - eXtreme Gradient Boosting (Tree) library
# https://github.com/tqchen/xgboost/wiki/Parameters
# max.depth - maximum depth of the tree
# nrounds - the max number of iterations

# take first 10% of the data only!
trainPortion <- floor(nrow(adultsTrsf)*0.1)

trainSet <- adultsTrsf[ 1:floor(trainPortion/2),]
testSet <- adultsTrsf[(floor(trainPortion/2)+1):trainPortion,]

smallestError <- 100

for (depth in seq(1,10,1)) {
  for (rounds in seq(1,20,1)) {
    
    # train
    bst <- xgboost(data = as.matrix(trainSet[,predictors]),
                   label = trainSet[,outcomeName],
                   max.depth=depth, nround=rounds,
                   objective = "reg:linear", verbose=0)
    gc()
    
    # predict
    predictions <- predict(bst, as.matrix(testSet[,predictors]), outputmargin=TRUE)
    err <- rmse(as.numeric(testSet[,outcomeName]), as.numeric(predictions))
    
    if (err < smallestError) {
      smallestError = err
      print(paste(depth,rounds,err))
    }     
  }
}  




