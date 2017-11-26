#load libraries
library(dtplyr) #data wrangling
library('ggthemes') #visualization
library(ggplot2) #visualization
library(data.table)
library(randomForest) #random forest
library(e1071) #svm
library(rpart) #decision tree

#load train.csv
train <- read.csv('../input/train.csv', stringsAsFactors = FALSE)
#load test.csv
test  <- read.csv('../input/test.csv', stringsAsFactors = FALSE)

#combine twp data sets
test$Survived <- NA
allData <- rbind(train,test)

#show the column names
colnames(allData)

#show first few rows of the data
head(allData)

#check the structure of the data
str(allData)

#converting categorical variables to factors
categorical_variables <- c('Survived', 'Pclass', 'Sex', 'Embarked')
allData[categorical_variables] <- lapply(allData[categorical_variables], function(x) as.factor(x))

##################################
## Data Preparation
##################################

#extracting titles from Name
names <- allData$Name
titles <-  gsub("^.*, (.*?)\\..*$", "\\1", names)

#adding the column Titles in the dataset
allData$Titles <- titles
table(allData$Sex, allData$Title)

#replacing other titles to 'Others'
allData$Titles[allData$Titles != 'Mr' & allData$Titles != 'Miss' & allData$Titles != 'Mrs' & allData$Titles != 'Master'] <- 'Others'
table(allData$Sex, allData$Title)

#checking for columns with missing data
MissingData <- c()
for(column in names(allData)) {
    missing <- sapply(allData[[column]], function(x) {return(x=='' | is.na(x))})
    MissingData <- c(MissingData, sum(missing))
}
names(MissingData) <- names(allData)

## The missing values in Survived corresponds to the test data. We see that Cabin has so many missing rows, but we think that this is not an important variable anyway so we ignore it. We now fill in the missing data in Age, Fare and Embarked columns.

#find the person with missing fare
allData[is.na(allData$Fare),]

#How much do people of Pclass=3 pay?
thirdClass <- allData[allData$Pclass==3,]
print(paste("The median fare of PClass=3 passengers is:", median(thirdClass$Fare, na.rm=TRUE)))

#Fill in missing fare
allData$Fare[is.na(allData$Fare)] <- median(thirdClass$Fare, na.rm=TRUE)

#find the persons with missing embarkment
allData[allData$Embarked=='',]

# Get rid of our missing values
embarkFare <- allData[allData$Embarked!='',]

# Use ggplot2 to visualize embarkment, passenger class, & median fare
ggplot(embarkFare, aes(x = Embarked, y = Fare, fill = factor(Pclass))) +
  geom_boxplot() +
  geom_hline(aes(yintercept=80), 
    colour='orange', linetype='solid', lwd=1) +
  labs(fill="Passenger Class")
  
#fill in missing embarkment
allData$Embarked[allData$Embarked==''] <- 'C'

#Use stepwise selection for predictors for the linear model
allDataAgeLM <- lm(Age~Pclass+Sex+SibSp+Parch+Fare+Embarked+Titles, data=allData)
simplifiedAgeLM <- step(allDataAgeLM)
predictedAge <- allData$Age
predictedAge[is.na(predictedAge)] <- predict(simplifiedAgeLM, allData[is.na(allData$Age),])

#plotting age distributions
par(mfrow=c(1,2))
hist(allData$Age, freq=F, main='Original Age', 
  col='grey', ylim=c(0,0.04), breaks =20)
hist(predictedAge, freq=F, main='Predicted Age', 
  col='lightgreen', ylim=c(0,0.04), breaks =20)
  
#fill in missing ages
allData$Age <- predictedAge

##################################
## Data Visualization
##################################

#percentage of people surviving the disaster
print(paste("The percentage of people surviving the disaster: ", round(mean(train$Survived) * 100, 2)))

#Gender vs Survival
genderImpact <- data.table(table(allData[1:891, "Sex"], train$Survived))
names(genderImpact) <- c("Sex","Survived","Count")
genderImpact[, Percentage := sum(Count), by=list(Sex)]
genderImpact[, Percentage := Count/Percentage*100]
ggplot(genderImpact, aes(x=Sex, y=Percentage, fill=Survived)) + geom_histogram(stat = "identity")

#Pclass vs Survived
pclassImpact <- data.table(table(allData[1:891, "Pclass"], train$Survived))
names(pclassImpact) <- c("Pclass","Survived","Count")
pclassImpact[, Percentage := sum(Count), by=list(Pclass)]
pclassImpact[, Percentage := Count/Percentage*100]
ggplot(pclassImpact, aes(x=Pclass, y=Percentage, fill=Survived)) + geom_histogram(stat = "identity")

#Parch vs Survived
ParchImpact <- data.table(table(allData[1:891,"Parch"], train$Survived))
names(ParchImpact) <- c("Parch","Survived","Count")
ParchImpact[, Percentage := sum(Count), by=list(Parch)]
ParchImpact[, Percentage := Count/Percentage*100]
ggplot(ParchImpact, aes(x=Parch, y=Percentage, fill=Survived)) + geom_histogram(stat = "identity")

#Siblings vs Survived
SiblingImpact <- data.table(table(allData[1:891,"SibSp"], train$Survived))
names(SiblingImpact) <- c("SibSp","Survived","Count")
SiblingImpact[, Percentage := sum(Count), by=list(SibSp)]
SiblingImpact[, Percentage := Count/Percentage*100]
ggplot(SiblingImpact, aes(x=SibSp, y=Percentage, fill=Survived)) + geom_histogram(stat = "identity")

#Embarked vs Survived
EmbarkedImpact <- data.table(table(allData[1:891, "Embarked"], train$Survived))
names(EmbarkedImpact) <- c("Embarked","Survived","Count")
EmbarkedImpact[, Percentage := sum(Count), by=list(Embarked)]
EmbarkedImpact[, Percentage := Count/Percentage*100]
ggplot(EmbarkedImpact, aes(x=Embarked, y=Percentage, fill=Survived)) + geom_histogram(stat = "identity")

#Fare vs Survived
FareImpact <- data.frame(allData[1:891,"Fare"], train$Survived)
FareImpact[,1] <- cut(FareImpact[,1], 10)
FareImpact <- data.table(table(FareImpact))
names(FareImpact) <- c("Fare","Survived","Count")
FareImpact[, Percentage := sum(Count), by=list(Fare)]
FareImpact[, Percentage := Count/Percentage*100]
ggplot(FareImpact, aes(x=Fare, y=Percentage, fill=Survived)) + geom_histogram(stat = "identity")

#Age vs Survived
AgeImpact <- data.frame(allData[1:891, "Age"], train$Survived)
AgeImpact[,1] <- cut(AgeImpact[,1], 10)
AgeImpact <- data.table(table(AgeImpact))
names(AgeImpact) <- c("Age","Survived","Count")
AgeImpact[, Percentage := sum(Count), by=list(Age)]
AgeImpact[, Percentage := Count/Percentage*100]
ggplot(AgeImpact, aes(x=Age, y=Percentage, fill=Survived)) + geom_histogram(stat = "identity")

#Title vs Survived
TitleImpact <- data.table(table(allData[1:891, "Titles"], train$Survived))
names(TitleImpact) <- c("Titles","Survived","Count")
TitleImpact[, Percentage := sum(Count), by=list(Titles)]
TitleImpact[, Percentage := Count/Percentage*100]
ggplot(TitleImpact, aes(x=Titles, y=Percentage, fill=Survived)) + geom_histogram(stat = "identity")

##################################
## Data Modelling
##################################

# Spliting the data back to a train and a test set
train <- allData[1:891,]
test <- allData[892:1309, -2]

#logistic regression
logistic_model <- glm(factor(Survived) ~ Age + Fare + Sex + Embarked + Parch + SibSp 
                 + Titles + Pclass, data = train,family = binomial)

#predicted result
ans_logistic = rep(NA,891)
for(i in 1:891){
  ans_logistic[i] = round(logistic_model$fitted.values[[i]],0)
}

#check result
mean(ans_logistic == train$Survived)
table(ans_logistic)

#random forest
set.seed(123)
rf_model <- randomForest(factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked,
                                          data = train)
# predicted result of regression
rf.fitted = predict(rf_model)
ans_rf = rep(NA,891)
for(i in 1:891){
  ans_rf[i] = as.integer(rf.fitted[[i]]) - 1
}
    # check result
mean(ans_rf == train$Survived)
table(ans_rf)

# decision tree
dt_model <- rpart(factor(Survived) ~ Age + Fare + Sex + Embarked + Parch + SibSp 
                 + Titles + Pclass, data = train)

# predicted result of regression
dt.fitted = predict(dt_model)
ans_dt = rep(NA,891)
for(i in 1:891){
  if(dt.fitted[i,1] >= dt.fitted[i,2] ){
    ans_dt[i] = 0
  } else{
    ans_dt[i] = 1
  }
}
#check result
mean(ans_dt == train$Survived)
table(ans_dt)

# svm
svm_model <- svm(factor(Survived) ~ Age + Fare + Sex + Embarked + Parch + SibSp 
                 + Titles + Pclass, data = train)

    # predicted result of regression
svm.fitted = predict(svm_model)
ans_svm = rep(NA,891)
for(i in 1:891){
  ans_svm[i] = as.integer(svm.fitted[[i]]) - 1
}
    # check result
mean(ans_svm == train$Survived)
table(ans_svm)

#logistic regression model
a = sum(ans_logistic ==1 & train$Survived == 1) #True Positive
b = sum(ans_logistic ==1 & train$Survived == 0) #False Positive
c = sum(ans_logistic ==0 & train$Survived == 1) #False Negative
d = sum(ans_logistic ==0 & train$Survived == 0) #True Negative

print(paste("The accuracy is ", round((a+d)/(a+b+c+d) * 100, 2),"%."))

#Random Forest model
a = sum(ans_rf ==1 & train$Survived == 1)
b = sum(ans_rf ==1 & train$Survived == 0)
c = sum(ans_rf ==0 & train$Survived == 1)
d = sum(ans_rf ==0 & train$Survived == 0)

print(paste("The accuracy is ", round((a+d)/(a+b+c+d) * 100, 2),"%."))

#Decision Tree model
a = sum(ans_dt ==1 & train$Survived == 1)
b = sum(ans_dt ==1 & train$Survived == 0)
c = sum(ans_dt ==0 & train$Survived == 1)
d = sum(ans_dt ==0 & train$Survived == 0)

print(paste("The accuracy is ", round((a+d)/(a+b+c+d) * 100, 2),"%."))

#SVM
a = sum(ans_svm ==1 & train$Survived == 1)
b = sum(ans_svm ==1 & train$Survived == 0)
c = sum(ans_svm ==0 & train$Survived == 1)
d = sum(ans_svm ==0 & train$Survived == 0)

print(paste("The accuracy is ", round((a+d)/(a+b+c+d) * 100, 2),"%."))

##################################
## Prediction of Survival
##################################

#prediction using logistic regression model
prediction_logistic = predict(logistic_model, newdata = test)
prediction_logistic = as.numeric(prediction_logistic > 0)
table(prediction_logistic)

#prediction using Random Forest model
prediction_rf = predict(rf_model, newdata = test)
prediction_rf = as.integer(prediction_rf) - 1
table(prediction_rf)

#prediction using decision tree model
prediction_dt = predict(dt_model, newdata = test)
prediction_dt = ifelse(prediction_dt[,1] >= prediction_dt[,2], 0, 1)
table(prediction_dt)

#prediction using svm
prediction_svm = predict(svm_model, newdata = test)
prediction_svm = as.integer(prediction_svm) - 1
table(prediction_svm)

# create a csv file for submittion
data<-data.frame(PassengerId = test$PassengerId, Survived = prediction_logistic)
write.csv(data,file = "PredictionLogistic.csv",row.names = FALSE)

data<-data.frame(PassengerId = test$PassengerId, Survived = prediction_rf)
write.csv(data,file = "PredictionRF.csv",row.names = FALSE)

data<-data.frame(PassengerId = test$PassengerId, Survived = prediction_dt)
write.csv(data,file = "PredictionDT.csv",row.names = FALSE)

data<-data.frame(PassengerId = test$PassengerId, Survived = prediction_svm)
write.csv(data,file = "PredictionSVM.csv",row.names = FALSE)






