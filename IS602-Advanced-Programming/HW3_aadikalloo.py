import pandas as pd

def load_data(csv_path):
	data = pd.read_csv(csv_path)
	return data

def 

if __name__ == '__main__':
	filepath = "/Users/aadi/Google Drive/School/MS Data Analytics/IS602 Advanced Programming/Homework3/cars.data.csv"
	cars_data = load_data(fiepath)
