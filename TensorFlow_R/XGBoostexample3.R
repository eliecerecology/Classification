#http://xgboost.readthedocs.io/en/latest/R-package/discoverYourData.html
require(xgboost)
require(Matrix)
require(data.table)
if (!require('vcd')) install.packages('vcd')
require(vcd)
edit(Arthritis)
edit(df)

df <- data.table(Arthritis, keep.rownames = F)
head(df)
str(df)
head(df[,AgeDiscret := as.factor(round(Age/10,0))])
df$AgeDiscret
head(df[,AgeCat:= as.factor(ifelse(Age > 30, "Old", "Young"))])

df[,ID:=NULL]
levels(df[,Treatment])

sparse_matrix <- sparse.model.matrix(Improved~.-1, data = df)
dim(sparse_matrix)
edit(df)
output_vector = df[,Improved] == "Marked"
output_vector

bst <- xgboost(data = sparse_matrix, label = output_vector, max.depth = 4,
               eta = 1, nthread = 2, nround = 10,objective = "binary:logistic")

importance <- xgb.importance(feature_names = sparse_matrix@Dimnames[[2]], model = bst)
head(importance)

importanceRaw <- xgb.importance(feature_names = sparse_matrix@Dimnames[[2]], model = bst, data = sparse_matrix, label = output_vector)

# Cleaning for better display
importanceClean <- importanceRaw[,`:=`(Cover=NULL, Frequency=NULL)]

head(importanceClean)
xgb.plot.importance(importance_matrix = importanceRaw)
