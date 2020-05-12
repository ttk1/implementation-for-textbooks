#!/usr/bin/env python3

import os
import requests

# make data directory
dirpath = "./data"
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

# data_dict {filename:url}
data_dict = {
    "AirPassengers.csv": "https://www.analyticsvidhya.com/wp-content/uploads/2016/02/AirPassengers.csv",
    "m_quote.csv": "https://www.mizuhobank.co.jp/market/csv/m_quote.csv"
    }

# download function
def download(data_dict, dirpath):
    for filename, url in data_dict.items():
        filepath = os.path.join(dirpath, filename)
        if not os.path.exists(filepath): # download
            print("{} : start downloading".format(filename))
            try:
                content = requests.get(url).text
                with open(filepath, "w") as f:
                    f.write(content)
                print(" download completed")
            except:
                print("Error! Invalid URL? Please check data_dict.")
        else:
            print("{} : already exists".format(filename))
        
# execution
download(data_dict, dirpath)