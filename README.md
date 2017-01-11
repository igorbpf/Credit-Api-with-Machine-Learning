# Credit-Api-with-Machine-Learning
A credit analysis api powered by a machine learning algorithm

This project is an API for Credit Analysis. It is powered by a machine learning algorithm which decides whether the client
can get credit or not from the financial institution. It was inspired by Kaggle's competition "Give me some credit" 
https://www.kaggle.com/c/GiveMeSomeCredit 
 
 The file train_model.py set the classifier. 
 
 To start the API, run python app.py and browse localhost:5000
 
 Or instead check the API running in production at: http://mlbanking.herokuapp.com/
 
 Also possible send requests via command line by curl:
 
curl -X POST \

   --header "Content-Type: application/x-www-form-urlencoded" \
   
   --header "Accept: application/json" \
   
   -d "NumberOfOpenCreditLinesAndLoans=12" \
   
   -d "NumberRealEstateLoansOrLines=0" \
   
   -d "Age=24" \
   
   -d "DebtRatio=0.3" \
   
   -d "NumberOfDependents=0" \
   
   -d "MonthlyIncome=12500" \
   
   -d "RevolvingUtilizationOfUnsecuredLines=0.319779462" \
   
   -d "NumberOfTimes90DaysLate=12" \
   
   -d "NumberOfTime60-89DaysPastDueNotWorse=0" \
   
   -d "NumberOfTime30-59DaysPastDueNotWorse=0" \
   
   "https://mlbanking.herokuapp.com/approve_credit/"
