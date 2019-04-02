import requests	# request is most basic paser

page_address = "https://beomi.github.io/beomi.github.io_old"

## HTTP GET Request
req = requests.get(page_address)

## HTTP source
source = req.text
print(source)

## HTTP Header
header = req.headers
print(header)

## HTTP Status: 200 == OK
status = req.status_code
print(status)

## Check the normal of HTTP
is_ok = req.ok
print(is_ok)