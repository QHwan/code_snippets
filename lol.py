import requests
import json

key = 'RGAPI-72361b19-aa17-49fa-a94e-3e37ad7be1d2'
url = 'https://na1.api.riotgames.com/lol/summoner/v4/summoners/by-name/RiotSchmick?api_key=' + key

res = requests.get(url)

print(res.status_code)
print(res.headers)
print(res.json())