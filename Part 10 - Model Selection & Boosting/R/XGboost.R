# XGboost

# Importing the dataset
dataset = read.csv('Churn_Modelling.csv')
dataset = dataset[4:14]

# Encoding categorical data
dataset$Geography = as.numeric(factor(dataset$Geography,
																			levels = c('France', 'Spain', 'Germany'),
																			labels = c(1, 2, 3)))
dataset$Gender = as.numeric(factor(dataset$Gender,
																	 levels = c('Female', 'Male'),
																	 labels = c(1, 2)))


# Splitting dataset into training set and test set
library(caTools)
set.seed(123)
split = sample.split(dataset$Exited, SplitRatio = 0.80)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Applying XGboost to the training set
# install.packages('xgboost')
library(xgboost)
classifier = xgboost(data = as.matrix(training_set[-11]),
										 label = training_set$Exited,
										 nrounds = 10)


# Applying k-fold CrossValidation
# install.packages('caret')
library(caret)
folds = createFolds(training_set$Exited, k = 10)
cv = lapply(folds, function(x) {
	training_fold = training_set[-x, ]
	test_fold = training_set[x, ]
	classifier = xgboost(data = as.matrix(training_set[-11]),
											 label = training_set$Exited,
											 nrounds = 10)
	y_pred = predict(classifier, newdata = as.matrix(test_fold[-11]) )
	y_pred = (y_pred >= 0.5)
	cm = table(test_fold[, 11], y_pred)
	accuracy = (cm[1, 1] + cm[2, 2]) / (cm[1, 1] + cm[1, 2] + cm[2, 1] + cm[2, 2])
	return(accuracy)
	
})
accuracy = mean(as.numeric(cv))
