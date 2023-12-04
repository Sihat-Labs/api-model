import requests
import json

url = "http://127.0.0.1:5000"

# select slice
data = {'n_slice': 25}

files = [("file", ("t1.nii", open("t1.nii", "rb"))),
         ("file", ("t1ce.nii", open("t1ce.nii", "rb"))),
         #  ("file", ("t2.nii", open("t2.nii", "rb"))),
         #  ("file", ("flair.nii", open("flair.nii", "rb")))
         ]

response = requests.post(url, files=files, data=data)

if response.status_code == 200:
    result = json.loads(response.content)
    image_base64 = result.get("image_base64")
    print("Received image base64:", image_base64)
else:
    print("Error:", response.text)
