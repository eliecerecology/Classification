rm(list=ls(all=TRUE))
#https://www.r-bloggers.com/an-introduction-to-xgboost-r-package/
#https://github.com/sachinruk/xgboost_tut/blob/master/xg_tut.R
https://www.youtube.com/watch?v=87xRqEAx6CY

install.packages("DiagrammeR")
library(DiagrammeR)
library(xgboost) # model
require(xgboost)
set.seed(1)
data(agaricus.train, package='xgboost')
data(agaricus.test, package='xgboost')
edit(agaricus.train$data)


train <- agaricus.train
test  <- agaricus.test
edit(train)
edit(as.matrix(train$data))
param <- list("objective" = "binary:logistic",
              "eval_metric" = "logloss",
              "eta" = 1, "max.depth" = 2)
bst.cv = xgb.cv(param=param, data = as.matrix(train$data), label = train$label, nfold = 10, nrounds = 20)
bst.cv$evaluation_log$test_logloss_mean
plot(log(bst.cv$evaluation_log$test_logloss_mean), type = "l")
bst <- xgboost(data = as.matrix(train$data), label = train$label, max.depth = 2, eta = 1, nround = 5,
               nthread = 5, objective = "binary:logistic")

preds=predict(bst,test$data)
print(-mean(log(preds)*test$label+log(1-preds)*(1-test$label)))
trees = xgb.model.dt.tree(dimnames(train$data)[[2]],model = bst)

# Get the feature real names
names <- dimnames(train$data)[[2]]
importance_matrix <- xgb.importance(names, model = bst)
xgb.plot.importance(importance_matrix[1:10])
xgb.plot.tree(feature_names = names, model = bst, n_first_tree = 2)
