import requests
#response = requests.get('https://api.github.com/this-api-should-not-exist')
#response = requests.get('https://api.github.com')
#response = requests.get(
#    'https://api.github.com/search/repositories',
#    params={'q': 'requests+language:python'},
#)

response = requests.get('https://imgs.xkcd.com/comics/making_progress.png')

if response.status_code == 200:
    print('Success!')
elif response.status_code == 404:
    print('Not Found.')

with open(r'img.png','wb') as f:
    f.write(response.content)

print(response.content)
#print(response.json())

