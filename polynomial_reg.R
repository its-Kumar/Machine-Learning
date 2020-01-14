# importing dataset

dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

#fitting linear reg to dataset
lin_reg = lm(formula = Salary~., data = dataset)
summary(lin_reg)

# fitting polynomial reg to dataset
dataset$Level2 = dataset$Level ^2
dataset$Level3 = dataset$Level ^3
dataset$Level4 = dataset$Level ^4
poly_reg = lm(formula = Salary~.,data = dataset)

summary(poly_reg)

#Visualizing linear reg results
library(ggplot2)
ggplot() + geom_point(aes(x=dataset$Level , y=dataset$Salary ), colour = "red") + 
  geom_line(aes(x= dataset$Level, y=predict(lin_reg,newdata = dataset)), colour = "blue") +
  ggtitle("Truth or Bluff ( linear reg. )") +
  xlab("Lavels") + ylab("Salary")

#Visualizing polynomial reg results
ggplot() + geom_point(aes(x=dataset$Level , y=dataset$Salary ), colour = "red") + 
  geom_line(aes(x= dataset$Level, y=predict(poly_reg,newdata = dataset)), colour = "blue") +
  ggtitle("Truth or Bluff ( Polynomial reg. )") +
  xlab("Lavels") + ylab("Salary")
  
# predicting new result with linear reg
y_pred = predict(lin_reg, data.frame(Level = 6.5))


# predicting new result with polynomial reg
y_pred = predict(poly_reg, data.frame(Level = 6.5,Level2=6.5 ^2, Level3 = 6.5 ^3, Level4 = 6.5 ^4))

