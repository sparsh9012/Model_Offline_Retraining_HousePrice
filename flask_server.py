from flask import Flask, request, render_template, jsonify, make_response, request
import pandas as pd
import os
from sklearn.externals import joblib
import numpy as np

app = Flask(__name__)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
	if request.method == 'POST':
		new_data = pd.read_csv(request.files.get('file'))
		test_data = pd.read_csv('test_data.csv')

		testY_oldmodel = test_data['PRICE'].values
		testX_oldmodel = test_data.drop('PRICE', axis=1)

		old_model = joblib.load('oldmodel.pkl')
		predictions = old_model.predict(testX_oldmodel)
		from sklearn import metrics
		mae_before = metrics.mean_absolute_error(testY_oldmodel, predictions)
		mse_before = metrics.mean_squared_error(testY_oldmodel, predictions)
		rmse_before = np.sqrt(metrics.mean_squared_error(testY_oldmodel, predictions))
		rsq_before = metrics.r2_score(testY_oldmodel, predictions)
				
		old_data = pd.read_csv('old.csv')
		#new_data = pd.read_csv(request.files.get('file')) #From user (through web-interface)
		df = pd.concat([old_data, new_data])

		removable_features = ['SOURCE', 'CENSUS_TRACT', 'CENSUS_BLOCK', 'SQUARE', 'USECODE', 'Unnamed: 0', 'X', 'Y']
		df = df.drop(removable_features, axis=1)

		df['SALEDATE'] = pd.to_datetime(df['SALEDATE'])
		df['SALE_YR'] = df['SALEDATE'].dt.year; df['SALE_MONTH'] = df['SALEDATE'].dt.month
		df = df.drop('SALEDATE', axis=1)
		df = df.drop('GIS_LAST_MOD_DTTM', axis=1)
		df = df[~pd.isnull(df['PRICE'])]
		df = df.drop(df[df.PRICE>2000000].index)

		df = df.loc[:, df.isnull().sum() < 0.4*df.shape[0]]
		df = df.drop('ASSESSMENT_SUBNBHD', axis=1)
		df = df.dropna(how='any')
		df = df.drop(['LATITUDE','LONGITUDE','ZIPCODE'], axis=1)

		df = df[df.FIREPLACES<=11]
		df = df[df.AYB>=1825]
		df = df[df.EYB>=1900]
		df = df[df.LANDAREA<=50000]

		df = pd.concat([df, pd.get_dummies(df[['AC','QUALIFIED','WARD','QUADRANT']])], axis=1)
		df = df.drop(['AC','QUALIFIED','WARD','QUADRANT'], axis=1)
		from sklearn.preprocessing import LabelEncoder
		le = LabelEncoder()
		df['HEAT']=le.fit_transform(df['HEAT'])
		df['ASSESSMENT_NBHD']=le.fit_transform(df['ASSESSMENT_NBHD'])

		df["PRICE"] = np.log1p(df["PRICE"])
		df = df.drop(df[df.PRICE<10].index)
		y = df['PRICE']; X = df.drop('PRICE', axis=1)
		
		from sklearn.model_selection import train_test_split
		X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20)
		from sklearn.ensemble import RandomForestRegressor
		forest_reg_new = RandomForestRegressor(random_state=42)
		forest_reg_new.fit(X_train, y_train)
		predictions = forest_reg_new.predict(X_test)

		from sklearn import metrics
		mae_after = metrics.mean_absolute_error(y_test, predictions)
		mse_after = metrics.mean_squared_error(y_test, predictions)
		rmse_after = np.sqrt(metrics.mean_squared_error(y_test, predictions))
		rsq_after = metrics.r2_score(y_test, predictions)
		
		x = ['', 'Mean Absolute Error', 'Mean Square Error', 'Root Mean Square Error', 'R-Square Value']
		before_retraining = ['Before Re-training', np.round(mae_before,4), np.round(mse_before,4), np.round(rmse_before,4), np.round(rsq_before,4)]
		after_retraining = ['After Re-training', np.round(mae_after,4), np.round(mse_after,4), np.round(rmse_after,4), np.round(rsq_after,4)]
		result = dict(zip(x, zip(before_retraining, after_retraining)))
		Table = []
		for key, value in result.items():
			temp = []
			temp.extend([key,value])  #Note that this will change depending on the structure of your dictionary
			Table.append(temp)
		Table
		
		return render_template('upload.html', table=Table)
	return render_template('upload.html')

if __name__ == '__main__':
	app.run(debug=True)