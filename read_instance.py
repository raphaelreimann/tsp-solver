
import sys


# comments
def readf(_file):
	# open instance file
	try:
		ifile = open(_file,'r')
	except:
		print("Abort: file '{}' doesn't exist".format(_file))
		sys.exit()
	
	n = 0
	# read instance line by line
	raw = ifile.readlines()
	ifile.close()
	
	# print("calling readf (fill me!)")
	drop_off_punkte = []

	for line in raw:
		drop_off_punkte.append(line.strip().split())
		n += 1
	
	return n, drop_off_punkte
