import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display


data = pd.read_csv("housing.csv")
#display(data)

# let's deal with null values, and reclassify/preprocess text entries in 'ocean_proximity' column

data.dropna(inplace = True) 

# now we prepare training and evaluation data (split the dataset)
# and also split X and Y

from sklearn.model_selection import train_test_split

X = data.drop(['median_house_value'], axis = 1)
Y = data['median_house_value']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# now we join the X and Y training data

train_data = X_train.join(Y_train)


#display(train_data)

train_data['total_rooms'] = np.log(train_data['total_rooms'] + 1)
train_data['total_bedrooms'] = np.log(train_data['total_bedrooms'] + 1)
train_data['population'] = np.log(train_data['population'] + 1)
train_data['households'] = np.log(train_data['households'] + 1)

train_data = train_data.join(pd.get_dummies(train_data.ocean_proximity)).drop(['ocean_proximity'], axis = 1)

if 'ISLAND' in train_data.columns:
    train_data.rename(columns={'ISLAND': 'INLAND'}, inplace=True)

train_data['bedroom_ratio'] = train_data['total_bedrooms'] / train_data['total_rooms'] #inverse rooms per bedroom
train_data['household_rooms'] = train_data['total_rooms'] / train_data['households'] #rooms per household

#plt.figure(1)
#train_data.hist(figsize = (15,8)) # quick view on the data
#plt.savefig('histogram')
#plt.show()

fig_data = train_data.iloc[:,:10]

fig_data.columns = ['Longitude','Latitude','Local Median\nAge','Total\nRooms','Total\nBedrooms','Block\nPopulation','Block\nHouseholds','Median\nIncome','Median House\nValue','Ocean\nProximity']

#sns.set(font_scale=1.2, style="whitegrid", font="Arial")

e = 1
if e == 1:

    sns.set(rc={"axes.facecolor": "#F0F0F0", "figure.facecolor": "#D8D8D8"})

    plt.figure(2, figsize = (20,16))
    sns.heatmap(fig_data.corr(), annot=True, cmap='YlGnBu')  # quick view of correlations between metrics
    plt.xticks(fontsize=16, rotation = 90)
    plt.yticks(fontsize=16, rotation = 0)
    plt.savefig('heatmap')
#plt.show()
# could drop features with near 0 correlation to house value to simplify the model

d = 1
if d == 1:
    plt.figure(3, figsize = (20,16))
    scatter_plot = sns.scatterplot(x="latitude", y="longitude", data=train_data, hue="median_house_value", palette="coolwarm")
    plt.xlabel('Latitude', fontsize = 20)
    plt.ylabel('Longitude', fontsize = 20)
    scatter_plot.legend(prop={'size': 15}, title='House Value', fontsize=14)
    scatter_plot.grid(False)
    scatter_plot.legend_.get_title().set_fontsize(15) 
    scatter_plot.tick_params(labelsize=15)
    plt.title('California House Prices', fontsize = 20)
    plt.savefig('scatter')
#plt.show()
#coastal houses are more expensive

# linear regression 

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train, Y_train = train_data.drop(['median_house_value'], axis = 1), train_data['median_house_value']
X_train_s = scaler.fit_transform(X_train)

reg = LinearRegression()

reg.fit(X_train_s, Y_train)  # fit our metrics to house value

test_data = X_test.join(Y_test)

test_data['total_rooms'] = np.log(test_data['total_rooms'] + 1)
test_data['total_bedrooms'] = np.log(test_data['total_bedrooms'] + 1)
test_data['population'] = np.log(test_data['population'] + 1)
test_data['households'] = np.log(test_data['households'] + 1)

test_data = test_data.join(pd.get_dummies(test_data.ocean_proximity)).drop(['ocean_proximity'], axis=1)
if 'ISLAND' in test_data.columns:
    test_data.rename(columns={'ISLAND': 'INLAND'}, inplace=True)

test_data['bedroom_ratio'] = test_data['total_bedrooms'] / test_data['total_rooms'] 
test_data['household_rooms'] = test_data['total_rooms'] / test_data['households'] 

X_test, Y_test = test_data.drop(['median_house_value'], axis = 1), test_data['median_house_value']
X_test_s = scaler.transform(X_test)

a = reg.score(X_test_s, Y_test)   # test our fit on untrained data to predict house value (score = 0.67)

print(a)

# now lets do random forest regressor + hyperparameter tuning

from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor()

forest.fit(X_train_s, Y_train)

b = forest.score(X_test_s, Y_test)

print(b)

from sklearn.model_selection import GridSearchCV

forest = RandomForestRegressor()

param_grid = {
    "n_estimators": [3, 10, 30],  # this is lower than the default on previous test
    "max_features": [2, 4, 6, 8]
}

grid_search = GridSearchCV(forest, param_grid, cv=5, scoring = "neg_mean_squared_error", return_train_score=True)

grid_search.fit(X_train_s, Y_train)

best_forest = grid_search.best_estimator_
c = best_forest.score(X_test_s, Y_test)

print(c)