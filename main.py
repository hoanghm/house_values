import pandas as pd
import numpy as np
import os 

'''
DataFrame slector
'''
from sklearn.base import BaseEstimator, TransformerMixin
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, isNumerical):
        self.isNumerical = isNumerical
        self.num_attribs = ['LotFrontage','LotArea','OverallQual','OverallCond','YearBuilt',
               'YearRemodAdd','MasVnrArea','BsmtFinSF1','BsmtUnfSF','TotalBsmtSF','1stFlrSF',
               '2ndFlrSF','GrLivArea','BsmtFullBath','FullBath','HalfBath','BedroomAbvGr',
               'TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars','GarageCars','GarageArea',
               'WoodDeckSF','OpenPorchSF','EnclosedPorch','ScreenPorch','PoolArea']
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if self.isNumerical:
            return X[self.num_attribs]
        else:
            return X.drop(self.num_attribs, axis=1)
       
        
'''
 Label Binarizer
'''
from sklearn.preprocessing import LabelBinarizer
class MyLabelBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        self.columns = [0] * X.shape[1]
        self.encoders = [0] * X.shape[1]
        for i in range(X.shape[1]):
            self.encoders[i]= LabelBinarizer(sparse_output=False).fit(X.iloc[:,i])
        return self
    def transform(self, X):
        for i in range(X.shape[1]):
            self.columns[i] = self.encoders[i].transform(X.iloc[:,i])
        result = tuple(self.columns)
        return np.c_[result]
    
'''
My Categorical Imputer
'''
class MyCatImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.fillna('NA')

'''
 My Label Encoder
'''
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
class MyLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = LabelEncoder()
    def fit(self, X, y=None):
        return self.encoder.fit(X)
    def transform(self, X):
        return self.encoder.transform(X).reshape(-1,1)
    
'''
 Variable Selector 
'''
def variableSelector(X):
    return X.drop(['3SsnPorch', 'BsmtFinSF2', 'BsmtHalfBath', 'MoSold', 'Id', 'LowQualFinSF', 
                          'YrSold', 'MiscVal', 'Alley'],
                         axis = 1)
    
        

data = pd.read_csv('train.csv')

data = variableSelector(data)

# Split the data into train set and test set 
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
X = train_set.drop('SalePrice', axis=1)
y = train_set['SalePrice'].copy()
test_X = test_set.drop('SalePrice', axis=1)
test_y = test_set['SalePrice'].copy()

'''
 Create a Pipeline
'''
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Imputer, StandardScaler 

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(isNumerical=True)),
        ('imputer', Imputer(missing_values="NaN", strategy='median')),
        ('std_scaler', StandardScaler())
        ])

cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(isNumerical=False)),
        ('imputer', MyCatImputer()),
        ('label_Binarizer', MyLabelBinarizer())
        ])

full_pipeline = FeatureUnion(transformer_list=[
        ('num_pipeline', num_pipeline),
        ('cat_pipeline', cat_pipeline)
        ])
    

houses_prepared = full_pipeline.fit_transform(X)

'''
 Fit the model
'''
from sklearn.ensemble import RandomForestRegressor
predictor = RandomForestRegressor()
predictor.fit(houses_prepared, y)

'''
 Use the model on the actual test set
'''
initial_test_data = pd.read_csv('test.csv')
test_data = variableSelector(initial_test_data)


test_X_prepared = full_pipeline.transform(test_data)

predictions = predictor.predict(test_X_prepared)

results = np.c_[initial_test_data['Id'], predictions]

np.savetxt(fname='result.csv', X=results, delimiter=',')

        


