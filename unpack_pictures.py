import json
import urllib.request

with open('data.json') as file_name:
    data = json.load(file_name)

for i in range(0, len(data['results'])):
    try:
        picture_URL = data['results'][i]['officialPictureUrl']
        name = data['results'][i]['displayName']
        with urllib.request.urlopen(picture_URL) as url:
            with open('Mugshots/{}.jpg'.format(name), 'wb') as f:
                f.write(url.read())

    except KeyError:
        pass
