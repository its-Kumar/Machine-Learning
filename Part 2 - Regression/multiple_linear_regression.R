# Multiple linear regression

# Importing the dataset
dataset = read.csv("50_Startups.csv")

# Encoding categorical data
dataset$State = factor(dataset$State, levels = c('New York','California','Florida'),
                       labels = c(1,2,3))


#install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset,split ==TRUE)
test_set = subset(dataset,split ==FALSE)

# fitting multi reg on training set
regressor = lm(formula = Profit~ . , data = training_set)

# Predicting the results
y_pred = predict(regressor, newdata = test_set)

# Backward elimination
regressor = lm(formula = Profit~R.D.Spend + Administration + Marketing.Spend + State , 
               data = dataset)
summary(regressor)

regressor = lm(formula = Profit~R.D.Spend + Administration + Marketing.Spend , 
               data = dataset)
summary(regressor)


regressor = lm(formula = Profit~R.D.Spend + Marketing.Spend , 
               data = dataset)
summary(regressor)

regressor = lm(formula = Profit~R.D.Spend, 
               data = dataset)
summary(regressor)
