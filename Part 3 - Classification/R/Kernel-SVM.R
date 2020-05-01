# Kernel - SVM 

# Importing the dataset
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[, 3:5]

# Splitting dataset into training set and test set
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature scaling
training_set[, 1:2] = scale(training_set[, 1:2])
test_set[, 1:2] = scale(test_set[, 1:2])


# fitting classifier to training set
# install.packages('e1071')
library(e1071)
classifier = svm(formula = Purchased ~ ., 
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'radial')

# Predicting the test set
y_pred = predict(classifier, newdata = test_set[-3])

# Making the confusion matrix
cm = table(test_set[, 3], y_pred)

# Visualise the training set results
#install.packages('ElemStatLearn')
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1] -1), max(set[, 1] + 1), by=0.01)
X2 = seq(min(set[, 2] -1), max(set[, 2] + 1), by=0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier,
                 type = "response",
                 newdata = grid_set
)
plot(set[, 3],
     main = "Kernel SVM Classifeir (Training Set)",
     xlab = "Age", ylab="Estimated Salary", 
     xlim=range(X1), ylim = range(X2))

contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch='.', col=ifelse(y_grid ==1, 'springgreen3', 'tomato') )
points(set, pch=21, bg=ifelse(set[, 3] ==1, 'green4', 'red3'))

# Visualise the test set results
#install.packages('ElemStatLearn')
library(ElemStatLearn)
set = test_set
X1 = seq(min(set[, 1] -1), max(set[, 1] + 1), by=0.01)
X2 = seq(min(set[, 2] -1), max(set[, 2] + 1), by=0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(classifier,
                 type = "response",
                 newdata = grid_set
)
plot(set[, 3],
     main = " SVM (Test Set)",
     xlab = "Age", ylab="Estimated Salary", 
     xlim=range(X1), ylim = range(X2))

contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch='.', col=ifelse(y_grid ==1, 'springgreen3', 'tomato') )
points(set, pch=21, bg=ifelse(set[, 3] ==1, 'green4', 'red3'))

