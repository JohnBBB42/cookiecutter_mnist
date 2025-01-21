import requests
pload = {'username':'Olivia','password':'123'}
response = requests.post('https://httpbin.org/post', data = pload)

print(response.content)
print(response.json())
