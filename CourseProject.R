library(caret)
library(e1071)
#library(ggplot2)
library(Hmisc)
library(randomForest)
#library(gbm)
#library(gridExtra)

# Set working directory
setwd("C:/Users/18662/Desktop/Data Science/8 - Machine Learning/Assignments")

#=======================================================================================
#                      Load and Clean Up Training and Test Data                        #
#=======================================================================================

# Load the CVS files from active directory
trainingdata = read.csv("pml-training.csv")
testingdata  = read.csv("pml-testing.csv")

# Clean training set to remove all patchy data and convert factor variables to factor type
training = trainingdata[,-c(1:7)]
toBeRemoved = which(training[1,]=="" | is.na(training[1,]))
training = training[,-toBeRemoved]
training$classe = as.factor(training$classe)

# Clean testing set to make it look like training set
testing = testingdata[,-c(1:7)]
testing = testing[,-toBeRemoved]
testing = testing[,1:52]

# Clean up memory
rm(trainingdata); rm(testingdata); rm(toBeRemoved)

#=======================================================================================
#                                  Random Forest Model                                 #
#=======================================================================================

# Create a random forest model using all variables, time it
ptm <- proc.time()
set.seed(123)
modFitRF = randomForest(classe ~ ., data=training, ntree=50) #OOB estimate of  error rate: 0.29%
modelingtime = proc.time() - ptm
# user  system elapsed 
#65.86    0.26   66.16 

# Predict the classe variable in the test set
predictionRF = predict(modFitRF, testing)

#=======================================================================================
#                         Generalized Boosted Regression Model                         #
#=======================================================================================

# Create a random forest model using all variables, time it
ptm <- proc.time()
set.seed(1234)
#modFitGBM = gbm(classe ~ ., data=training, n.trees=1000)
#modFitGBM = gbm.fit(x=training[,-53], y=training[,53], distribution="multinomial")

gbmGrid <-  expand.grid(interaction.depth = 1, n.trees = 100, shrinkage = 0.005)
fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 10)
modFitGBM = train(classe ~ ., data=training, trControl = fitControl, tuneGrid = gbmGrid)

modelingtime = proc.time() - ptm
# user  system elapsed 
#65.86    0.26   66.16 

# Predict the classe variable in the test set
predictionGBM = predict(modFitGBM, newdata=testing, n.trees=1000, type="response")

#=======================================================================================
#                   Cross validation using for RF random sampling                      #
#=======================================================================================

errors = vector()
ratioTestRows = 0.1 #Select ratio of rows to treat as cross-validation test set

for(i in 1:10){
  sampledRows = sample(nrow(training), round(nrow(training)*ratioTestRows), replace=FALSE)
  
  cv.testing = training[sampledRows,]
  cv.training = training[-sampledRows,]

  modFitRFCV = randomForest(classe ~ ., data=cv.training, ntree=50)
  
  predictionCV =predict(modFitRFCV, cv.testing)
  cm = confusionMatrix(cv.testing$classe, predictionCV)
  
  errors[i] = 1-cm$overall['Accuracy']
  print(paste("Error for pass", i, ":", errors[i]))
  
  rm(cv.training); rm(cv.testing); rm(modFitRFCV);
}

print(paste("Average error", ":", mean(errors)))

# "Error for pass 1  : 0.00458715596330272"
# "Error for pass 2  : 0.00356778797145774"
# "Error for pass 3  : 0.00407747196738018"
# "Error for pass 4  : 0.00254841997961264"
# "Error for pass 5  : 0.0010193679918451"
# "Error for pass 6  : 0.00203873598369009"
# "Error for pass 7  : 0.00407747196738018"
# "Error for pass 8  : 0.00458715596330272"
# "Error for pass 9  : 0.00254841997961264"
# "Error for pass 10 : 0.00305810397553519"
# 
# "Average error     : 0.00321100917431192"

#=======================================================================================
#                             Cross validation using GBM                               #
#=======================================================================================

errors = vector()
ratioTestRows = 0.1 #Select ratio of rows to treat as cross-validation test set

for(i in 1:3){
  sampledRows = sample(nrow(training), round(nrow(training)*ratioTestRows), replace=FALSE)
  
  cv.testing = training[sampledRows,]
  cv.training = training[-sampledRows,]
  
  modFitGBMCV = gbm(classe ~ ., data=cv.training)
  
  predictionCV =predict(modFitGBMCV, cv.testing)
  cm = confusionMatrix(cv.testing$classe, predictionCV)
  
  errors[i] = 1-cm$overall['Accuracy']
  print(paste("Error for pass", i, ":", errors[i]))
  
  rm(cv.training); rm(cv.testing); rm(modFitGBMCV);
}

print(paste("Average error", ":", mean(errors)))

#=======================================================================================
# Identify the importance of each variable
impFitRF = importance(modFitRF)[order(importance(modFitRF)[,1]),1]
names(impFitRF[impFitRF<127])

# Removing the least
trainingCleaner = training[ , -which(names(training) %in% names(impFitRF[impFitRF<127]))]

ptm <- proc.time()
modFitRFCleaner = randomForest(classe ~ ., data=trainingCleaner)
#OOB estimate of  error rate: 0.33%
#Confusion matrix:
#     A    B    C    D    E  class.error
#A 5577    2    0    0    1 0.0005376344
#B    9 3783    5    0    0 0.0036871214
#C    0   11 3406    5    0 0.0046756283
#D    0    0   20 3194    2 0.0068407960
#E    0    0    1    8 3598 0.0024951483
proc.time() - ptm
# user  system elapsed 
#54.79    0.38   55.28




# Try a tree model
modFitTrees = train(classe ~ ., method="rpart", data=training)
predictTrees = predict(modFitTrees, newdata=training) 
sum(predictTrees==training$classe) # 11086 correct
sum(predictTrees!=training$classe) # 8536 incorrect

# See how many elements are correlated
M = abs(cor(training[,-53]))
diag(M) = 0
Mcor = which(M>0.9, arr.ind=T)
Mcor = Mcor[order(Mcor[,1]),]
McorNames = cbind(names(M[1,][Mcor[,1]]), names(M[2,][Mcor[,2]]))

# Preprocess with PCA
preProc = preProcess(training[,-53], method="pca")
trainPC = predict(preProc, training[,-53])
modelFitPC = train(training$classe ~ ., method="rpart", data=trainPC)

trainingTotals = training[,c("total_accel_belt", "total_accel_arm", "total_accel_dumbbell", "total_accel_forearm", "classe")]
trainingTotals$classe = droplevels(trainingTotals$classe) # Remove unused levels of classe factor variable (A, D, E)
modFitRFTotals = randomForest(classe ~ ., data=trainingTotals)
predictRF = predict(modFitRF, newdata=trainingTotals) 

# Plot count of each outcome for each user
As = sum(training$classe=="A")
Bs = sum(training$classe=="B")
Cs = sum(training$classe=="C")
Ds = sum(training$classe=="D")
Es = sum(training$classe=="E")
qplot(unique(training$classe), c(As,Bs,Cs,Ds,Es), geom="bar")

# Plot dumbbell roll, pitch and yaw vs index, colored by 

qp1 = qplot(training[training$classe=="A", 55])
qp2 = qplot(training[training$user_name=="carlitos", 55])
qp3 = qplot(training[training$user_name=="charles", 55])
qp4 = qplot(training[training$user_name=="eurico", 55])
qp5 = qplot(training[training$user_name=="jeremy", 55])
qp6 = qplot(training[training$user_name=="pedro", 55])
grid.arrange(qp1, qp2, qp3, qp4, qp5, qp6, ncol=2)

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}