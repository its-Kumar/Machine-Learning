# Apriori

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
itemFrequencyPlot(dataset, topN = 10)


# Training Apriori on the dataset
# support = 3*7 /7500
# rules = apriori(data = dataset,
#								parameter = list(support = 0.003,																								 confidence = 0.2))
rules = apriori(data = dataset,
								parameter = list(support = 0.004,																								 confidence = 0.2))
# Visualising the results
inspect(sort(rules, by = 'lift')[1:10])
