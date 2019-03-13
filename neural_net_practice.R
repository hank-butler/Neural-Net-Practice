# Building a NN using UCI Bank Authentication Data Set

df <- read.csv("bank_note_data.csv")

head(df)

str(df)

# Exploratory Data Analysis

ggplot(data = df, aes(x = Image.Var, y = Image.Skew))+
  geom_point()

ggplot(data = df, aes(x = Image.Var, y = Image.Curt))+
  geom_point()

# Train Test Split

library(caTools)

set.seed(101)

split = sample.split(df$Class, SplitRatio = 0.70)

train <- subset(df, split == T)
test <- subset(df, split == F)

# Check stucture of train data

str(train)

# Class does NOT need to be converted to factor since NN requires numeric information

# Building the NN

library(neuralnet)

nn <- neuralnet(Class ~ Image.Var + Image.Skew + Image.Curt + Entropy, 
                data = train,
                hidden = 10,
                linear.output = F)

# Predictions

predicted.nn.values <- compute(nn, test[,1:4])

# Check head, still are probabilities

head(predicted.nn.values$net.result)

# Apply round function so only 0's and 1's as predicted classes

predictions <- sapply(predicted.nn.values$net.result, round)

head(predictions)

# Use table to create a confusion matrix, predictions v real values
table(predictions, test$Class)

# predictions   0   1
#           0 229   0
#           1   0 183

# Result is a little suspicious since results are perfect and data wasn't scaled
# Compare this to a random forest
library(randomForest)

df$Class <- factor(df$Class)

set.seed(101)
split = sample.split(df$Class, SplitRatio = 0.70)

train = subset(df, split == TRUE)
test = subset(df, split == FALSE)

model <- randomForest(Class ~ Image.Var + Image.Skew + Image.Curt + Entropy,
                      data = train)

rf.model <- randomForest(Class ~ .,
                         data = train)

rf.pred <- predict(rf.model, test)

table(rf.pred, test$Class)

# Random forest performed very well, makes sense that NN performed better.