# Importing Dependencies
from support_functions import get_data
from support_functions import missing_value_treatment
from support_functions import data_standardization
from support_functions import cat_encoding
from support_functions import num_summary
from support_functions import class_balance
from support_functions import top_5_num_features
from support_functions import hist_box_plots
from support_functions import data_split_and_CV
from support_functions import models
from support_functions import knn_learn
from support_functions import model_eval





# Master Function
def master(data=None,X_train=None, X_test=None, y_train=None, y_test=None,sk_folds=None,knn=None,scoresCV=None):
    print("1- Fetch Data","2- Missing Values Treatment","3- EDA: Numerical Summary",
          "4- EDA: Class Balance(%)","5- EDA: Histogram & Box Plot of Important Numerical Features",
          "6- Categorical Encoding","7- Standardization of Numerical Features",
          "8- Outliers Detection & Treatment","9-  Data Splitting & Cross-Validation",
          "10- Model Selection","11- Model Training(KNN)", "12- Model Evaluation",
          "13- More Functions to be Added",sep='\n')
    sr=int(input("What do you want to do? "))
    
    if sr==1 and data==None:
        id=int(input("Please enter the UCI ID "))
        df=get_data(id)
        print("Successfuly Connected to the Data Source")
        return df
    
    elif sr==2:
        missing_value_treatment(data)
    
    elif sr==7:
        print("1- Standard Scaler","2- Min-Max Scaler", sep='\n')
        type=int(input("Which type of Scaler? "))
        data_standardization(data,type)
    
    elif sr==6:
        cat_encoding(data)
    
    elif sr==3:
        return num_summary(data)
    
    elif sr==4:
        return class_balance(data)
    
    elif sr==5:
        hist_box_plots(data)
    
    elif sr==8:
        return "Outlier detection and treatment functionality under progress!"
    
    elif sr==9:
        return data_split_and_CV(data)
    
    elif sr==10:
        models(X_train, X_test, y_train, y_test)
    
    elif sr==11:
        return knn_learn(data,X_train,y_train,X_test,sk_folds)
    
    elif sr==12:
        model_eval(knn,X_test,y_test,scoresCV)
    
    else:
        return "Please either select correct option or wait for future updates!"