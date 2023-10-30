# Read /data/options.json and print the value of the OPENAI_API_KEY key
import json
import os

with open('/data/options.json') as json_file:
    data = json.load(json_file)
    print(data['OPENAI_API_KEY'])
