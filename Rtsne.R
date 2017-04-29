rm(list=ls(all=TRUE))
install.packages("Rtsne")
## calling the installed package
setwd("C:/Users/localadmin_eliediaz/Documents/MEGA/CGIwork") #work
setwd("C:/Users/localadmin_eliediaz/Documents/MEGA/Storage/Cormorant2014/Jaime")
setwd("/home/worki/Documents/Storage/Cormorant2014/Jaime/")

train<- read.csv(file= "DATS.csv") ## Choose the train.csv file downloaded from the link above
train <- read.csv("tsne5.csv", 
         header = TRUE,
         dec = " ")
train <- read.table(file = "tsne4.txt",
                 header = TRUE,
                 dec = ".")
edit(train)
library(Rtsne)
## Curating the database for analysis with both t-SNE and PCA
Positions <-train$Position
train$Position<-as.factor(train$Position)

## for plotting
colors = rainbow(length(unique(train$Position)))
names(colors) = unique(train$Position)

## Executing the algorithm on curated data
tsne <- Rtsne(train[,-1], dims = 2, perplexity= 50, verbose=TRUE, max_iter = 1000000, check_duplicates=F)

exeTimeTsne<- system.time(Rtsne(train[,-1], dims = 2, perplexity=50, verbose=TRUE, max_iter = 1000000, check_duplicates=F))

## Plotting
plot(tsne$Y, t='n', main="tsne")
text(tsne$Y, labels=train$Position, col=colors[train$Position])
dim(train[,-1])
tsne
