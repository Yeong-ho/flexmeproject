from flask import Flask, request,jsonify,render_template
import json
test = '{ \
	"filename": "flexme25.json",\
	"id": "0x20",\
	"metadata": "{}",\
	"name": "flexme_token25",\
	"status": "Submitted",\
	"transactionHash": "0xd22958d7e730392a7113d7241c5b2a2aa807df50248b15c44facff87549feb9d",\
	"wallet": "0xF84ae1Bdd0a7b18d71c470C4d032d8E68bb32Cb2"\
}'

json_data= json.loads(test)

name =  json_data['name']
filename = name + '.json'
print(filename)
