#!/usr/bin/env python3

import os
import urllib.request

# make data directory
dirpath = "./data"
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

# data_dict {filename:url}
data_dict = {
    "AirPassengers.csv": "https://www.analyticsvidhya.com/wp-content/uploads/2016/02/AirPassengers.csv",
    'key2': 2, 
    'key3': 3}

# download function
def download(data_dict, dirpath):
    for filename, url in data_dict.items():
        filepath = os.path.join(dirpath, filename)
        if not os.path.exists(filepath): # download
            print("{} : not exist, start downloading".format(filename))
            
        else:
            print("{} : exist".format(filename))
        
# yee
download(data_dict, dirpath)
