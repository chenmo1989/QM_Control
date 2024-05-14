import datetime
import os
import glob

class DataLoggingHandle:
	def __init__(self):
		pass

	def save(self, expt_dataset, expt_name):
		# save base attributes
		expt_dataset.attrs['name'] = expt_name
		timestamp = datetime.datetime.now()
		# generate path, filename, save directory
		tPath = self.generate_save_path(timestamp)
		print(tPath)
		result_filepath = os.path.join(tPath, self.generate_filename(expt_dataset.attrs["name"], timestamp, tPath)) # without extension
		print(result_filepath)
		expt_dataset.attrs["directory"] = result_filepath
		# save data and json settings
		print('-'*10 + 'saved to ' + result_filepath) 
		expt_dataset.to_netcdf(result_filepath + '.nc')
		return expt_dataset

	def generate_filename(self, expt_prefix, timestamp, tPath):
		num_file = len(glob.glob(tPath + expt_prefix+'*'))
		date = '{}'.format(timestamp.strftime('%Y-%m-%d'))

		if num_file == 0:
			tFilename = "{}_{}".format(date, expt_prefix)
		else:
			tFilename = "{}_{}_{}".format(date, expt_prefix, num_file+1)

		return tFilename

	def generate_save_path(self, timestamp):
		year = timestamp.strftime("%Y")
		month = timestamp.strftime("%m")
		day = timestamp.strftime("%d")
		tPath = os.path.join(r'/Users/chenmo/Documents/GitHub/QM_control/xarray_test',year,month,'Data_'+month+day+'/')
		if not os.path.exists(tPath):
			os.makedirs(tPath)
		self.tPath = tPath
		return tPath