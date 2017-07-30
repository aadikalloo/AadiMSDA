from bs4 import BeautifulSoup as BS
import pandas as pd
import os
import urllib.request
import re
import joblib
num_cores = 8

dz_pdf = pd.read_csv('dzpdf.csv')
folder_names = dz_pdf['collection'].unique()

img_path = '/home/aadi_kalloo/cdsa_imgs/tiles/'

def process_download_tile(url_idx, df_slice, category, save_dir, x, y):
	img_filename = save_dir + df_slice['name'].iloc[url_idx] + '_' + str(x) + '_' + str(y) + ".jpg"
	img_url = df_slice['url'].iloc[url_idx]
	img_url = img_url[: -17]
	img_url = img_url + "_files/15/" + str(x) + '_' + str(y) + '.jpg'
	#print(img_url)
	try:
		urllib.request.urlretrieve(img_url, img_filename)
		fs = os.path.getsize(img_filename)
		if fs < 7000:
			os.remove(img_filename)
			print('deleted ' + img_filename + ' of size ' + str(fs))
	except:
		pass
	print(category + ': ' + str(url_idx + 1) + '/' + str(len(df_slice['url']) + 1) + ' -- ' + str(x) + ',' + str(y))

def main():
	for category in folder_names[2: 3]:
		category_dir = img_path + category
		print(category)
		if not os.path.exists(category_dir):
			os.makedirs(category_dir)
		df_slice = dz_pdf[dz_pdf['collection'].str.contains(category)]
		for url_idx in range(100, len(df_slice['url'])):
			save_dir = img_path + category + '/' + df_slice['name'].iloc[url_idx] + '/'
			if not os.path.exists(save_dir):
				os.makedirs(save_dir)# for x in range(0, 100):
			joblib.Parallel(n_jobs = 16)(joblib.delayed(process_download_tile)(url_idx, df_slice, category, save_dir, x, y) for y in range(0, 30) for x in range(0, 100))

if __name__ == '__main__':
	main()
