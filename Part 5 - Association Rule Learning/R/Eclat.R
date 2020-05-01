# Eclat

# Data Preprocessing
dataset = read.csv('Market_BAsket_Optimisation.csv',
									 header = FALSE)
# install.packages('arules')
library(Matrix)
library(arules)
dataset = read.transactions('Market_BAsket_Optimisation.csv',
														sep =',',
														rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 100)


# Training Eclat on the dataset
# support = 3*7 /7500
rules = eclat(data = dataset,
								parameter = list(support = 0.004, minlen=2))


# Visualising the results
inspect(sort(rules, by = 'support')[1:10])
