import requests	# request is most basic parser, only returns string data
from bs4 import BeautifulSoup # BS parses html -> meaningful python class structure
import json
import os

# location of python file: useful expression!
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
page_address = "https://beomi.github.io/beomi.github.io_old"

## HTTP GET Request
req = requests.get(page_address)

## HTTP source
source = req.text

## It converts text -> python class
soup = BeautifulSoup(source, 'html.parser')

## Find titles from CSS Selector: CSS grammar?
my_titles = soup.select( 'h3 > a') # my_titles is the list of instances

data = {}

for title in my_titles:
	# property of Tag(ex: href property)
	data[title.text] = title.get('href')

with open(os.path.join(BASE_DIR, 'result.json'), 'w+') as json_file:
	json.dump(data, json_file)



## HTTP Header
header = req.headers

## HTTP Status: 200 == OK
status = req.status_code

## Check the normal of HTTP
is_ok = req.ok