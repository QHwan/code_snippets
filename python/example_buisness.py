import requests
from bs4 import BeautifulSoup

response_obj = requests.get('https://en.wikipedia.org/wiki/Special_wards_of_Tokyo').text
soup = BeautifulSoup(response_obj, 'lxml')
wards_Tokyo_Table = soup.find('table', {'class': 'wikitable sortable'})

### preparation of the table
column_names = ['Name', 'Kanji', 'Pop', 'Density',
                'num', 'flag', 'Area', 'Major_District']
n_columns = len(column_names)
data = []
for i in range(n_columns):
    data.append([])

for row in wards_Tokyo_Table.findAll("tr"):
    Ward = row.findAll('td')
    if len(Ward) == n_columns:
        for i in range(n_columns):
            data[i].append(Ward[i].find(text=True))