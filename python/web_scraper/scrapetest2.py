from urllib.request import urlopen
from urllib.error import HTTPError
from bs4 import BeautifulSoup

def getTitle(url):
    try:
        html = urlopen(url)
    except HTTPError as e:
        return None

    try:
        bsObj = BeautifulSoup(html.read(), features="html.parser")
        title = bsObj.body.h1
    except AttributeError as e:
        return None
    
    return title

title = getTitle("http://pythonscraping.com/pages/warandpeace.html")
if title == None:
    print("Title could not be found")
else:
    print(title)

html = urlopen("http://pythonscraping.com/pages/warandpeace.html")
bsObj = BeautifulSoup(html, features="html.parser")
nameList = bsObj.findAll("span", {"class":"green"})
for name in nameList:
    print(name.get_text())