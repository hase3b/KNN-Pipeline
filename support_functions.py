# Importing Dependencies
import pandas as pd
import numpy as np
#pip install ucimlrepo in anaconda command line prompt
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import MinMaxScaler,StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from scipy.stats import skew, kurtosis, jarque_bera
import seaborn as sns
import matplotlib.pyplot as plt
#pip install lazypredict in anaconda command line prompt
from lazypredict.Supervised import LazyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_auc_score, plot_roc_curve

# Support Function

# Data Fetching Function
def get_data(id):
    data=fetch_ucirepo(id=id)
    X = data.data.features 
    y = data.data.targets
    df= pd.merge(X, y, left_index=True, right_index=True)
    return df, y.columns[0]

# Missing Values Treatment
def missing_value_treatment(df):
    missing_columns=(df.isna().sum()!=0)[(df.isna().sum()!=0)].index
    for i in missing_columns:
        if ((df[i].isna().sum()/len(df))<=0.05) & len(df)>30000:
            df[i].dropna(inplace=True)
            df.reset_index(inplace=True,drop=True)
        elif (df[i].isna().sum()/len(df))>0.5:
            df.drop([i],axis=1,inplace=True)
        elif (df[i].dtype=='object') or (df[i].dtype.name=='category'):
            mode_val = df[i].mode()[0]
            df[i].fillna(mode_val, inplace=True)
        else:
            skewness=skew(df[i].dropna())
            if abs(skewness)>1:
                df[i].fillna(df[i].median(), inplace=True)
            else:
                df[i].fillna(df[i].mean(), inplace=True)
    print("Dataset is treated for missing values successfully")

# Standardization of Numerical Columns
def data_standardization(df,type):
    num_var=[]
    for i in df.drop([label_column],axis=1).columns:
        if (df[i].dtype!='object') and (df[i].dtype.name!='category'):
            num_var.append(i)
    if type==1:
        scaler=StandardScaler()
        for i in num_var:
            df[i]=scaler.fit_transform(df[[i]]).round(2)
    elif type==2:
        scaler = MinMaxScaler()
        for i in num_var:
            df[i]=scaler.fit_transform(df[[i]]).round(2)

# Categorical Encoding
def cat_encoding(df):
    cat_var=[]
    for i in df.drop([label_column],axis=1).columns:
        if (df[i].dtype=='object') or (df[i].dtype.name=='category'):
            cat_var.append(i)
    label_encoder = LabelEncoder()
    for feature in cat_var:
        df[feature] = label_encoder.fit_transform(df[feature])

# Numerical Summary
def num_summary(df):
    return df.describe()

# Class Balance
def class_balance(df):
    return (df[label_column].value_counts(normalize=True)*100)

# Pick top 5 important numerical features
def top_5_num_features(df):
    num_var=[]
    for i in df.drop([label_column],axis=1).columns:
        if (df[i].dtype!='object') and (df[i].dtype.name!='category'):
            num_var.append(i)
    X_train=df[num_var]
    y_train=df[label_column]
    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(X_train, y_train)
    feature_importances = rf_classifier.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    top_5_numerical_features = feature_importance_df.head(5)['Feature'].tolist()
    return top_5_numerical_features

# Construct histogram and boxplot of important numerical features
def hist_box_plots(df):
    num_features=top_5_num_features(df)
    for i in num_features:
        print("")
        print(f"Box and Distribution Plot - {i}")
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=df[i])
        plt.title('Box Plot')
        plt.show()

        plt.figure(figsize=(8, 4))
        sns.histplot(data=df[i])
        plt.title('Distribution Plot')
        plt.show()

        skewness=skew(df[i])
        kurt=kurtosis(df[i])
        JB,p_val=jarque_bera(df[i])
    
        print("Skewness:",skewness)
        if skewness<0:
            print("Left Skewed",'\n')
        elif skewness>0:
            print("Right Skewed",'\n')
        else:
            print("Center",'\n')

        print("Kurtosis:",kurt)
        if kurt<3:
            print("Platykurtic",'\n')
        elif kurt>3:
            print("Leptokurtic",'\n')
        else:
            print("Mesokurtic",'\n')
    
        print("JB Test for Normality")
        print(JB,p_val)
        if p_val<0.05:
            print("Distribution is not normal",'\n')
        else:
            print("Distribution is normal",'\n')

# Data Split and CV
def data_split_and_CV(df):
    seed=int(input("Random State? "))
    test_size=int(input("What % of data to be left for test? "))/100
    k=int(input("How many folds required? "))
    X=df.drop([label_column],axis=1)
    y=df[label_column]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size,random_state=seed)
    
    sk_folds=StratifiedKFold(n_splits=k,random_state=seed,shuffle=True)
    
    return X_train,X_test,y_train,y_test,sk_folds

# Model Selection
def models(X_train, X_test, y_train, y_test):
    clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
    models,predictions = clf.fit(X_train, X_test, y_train, y_test)
    print(models)
    
# Model Training
def knn_learn(df,X_train,y_train,X_test,sk_folds):
    k=int(input("What is the value of K? "))
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    scoresCV=cross_val_score(knn, df.drop([label_column],axis=1), df[label_column], cv = sk_folds)
    return knn, scoresCV

# Model Evaluation
def model_eval(knn,X_test,y_test,scoresCV):
    y_prob=knn.predict_proba(X_test)
    y_pred_test=knn.predict(X_test)
    auc=roc_auc_score(y_test, y_prob[:, 1])
    class_report=classification_report(y_test, y_pred_test, output_dict=True)
    
    metrics_dict = {
        'precision': class_report['macro avg']['precision'],
        'recall': class_report['macro avg']['recall'],
        'f1_score': class_report['macro avg']['f1-score'],
        'accuracy': class_report['accuracy'],
        'auc': auc
    }
    
    print("")
    print("")
    print("")
    print ("Train-Test Split Evaluation")
    for metric, value in metrics_dict.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    
    plot_roc_curve(knn, X_test, y_test)
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()
    
    print("")
    print("")
    print("")
    print("Cross-Validation Evaluation")
    print("Cross Validation Scores: ", scoresCV)
    print("Average CV Score: ", scoresCV.mean())
    print("Number of CV Scores used in Average: ", len(scoresCV))