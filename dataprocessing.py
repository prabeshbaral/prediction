import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

#load data
df = pd.read_csv('dataset/housing.csv')
df.dropna(inplace=True)

df['total_rooms'] = np.log(df['total_rooms']+1)
df['total_bedrooms'] = np.log(df['total_bedrooms']+1)
df['population'] = np.log(df['population']+1)
df['households'] = np.log(df['households']+1)

df1=df.join(pd.get_dummies(df.ocean_proximity).astype(int)).drop(['ocean_proximity'],axis=1)

#adding bed_ratio and Household_room
df1['bedroom_ratio']= df1['total_bedrooms']/df1['total_rooms']
df1['household_room']=df1['total_rooms']/df1['households']

X= df1.drop(['median_house_value'] ,axis=1)
y= df1['median_house_value']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = RandomForestRegressor()

model.fit(X_train,y_train)

def calculate_price(features):
    
    predicted_value = model.predict(features)
    
    # Return the predicted value (as a single number)
    return predicted_value[0]
