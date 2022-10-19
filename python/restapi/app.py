from flask import request,jsonify, render_template

from werkzeug.utils import secure_filename
from flask_cors import CORS
from aioflask import Flask, request, Response
from mysql  import SelectWallet
import json
from getnfts import getNftList,getNftMetadata
import create_nft
import logging
import datetime

log = logging.getLogger()
log.setLevel(logging.INFO)
logfilename = datetime.datetime.now().strftime("./log/nftlog.txt")
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler(logfilename)
file_handler.setFormatter(formatter)
log.addHandler(file_handler)

#app settings
app = Flask(__name__)
#CORS(app)
@app.route('/submit', methods=['POST'])
async def submit():  

    data = request.form['data']
    f = request.files['file']
    f.save('./files/' + secure_filename(f.filename))
    path = './files/'+secure_filename(f.filename)
   
    result = await create_nft.main(path,data)
    
    data = json.loads(data)
    log.info('%s\n %s' %('submit',request.form))
    log.info('%s%s %s'%('nftname:',data['name'],result))
    
    return result


@app.route('/selectwallet', methods=['POST'])
async def create():
    
    userId = request.get_json()['id']
    userName = request.get_json()['name']
    data = SelectWallet(userId,userName)
    
    log.info('selectwallet \n%s%s %s'%('username:',userName, data))

    return data

@app.route('/getnfts', methods=['POST'])
async def getnft():
    
    userId = request.get_json()['uid']
    nftlist = getNftList(userId)
    log.info('getnfts \n%s%s %s'%('uid:',userId, nftlist))
   
    return nftlist


@app.route('/getmetadata', methods=['POST'])
async def getmetadata():
    
    userId = request.get_json()['uid']
    tokenId = request.get_json()['tokenid']
    metadata = getNftMetadata(userId,tokenId)
    log.info('getmeta \n%s%s %s'%('tokenId:',tokenId, metadata))
   
    return metadata



if __name__ == "__main__":
    app.run(host= '0.0.0.0', port='6000',debug=True)
