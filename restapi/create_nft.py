from statistics import mean
from flask import Flask, request,jsonify,render_template
import json
import os
import asyncio
import time
from mysql import NftInfoAdd

image_uploader = '''curl --location --request POST 'https://metadata-api.klaytnapi.com/v1/metadata/asset' --header 'x-chain-id: 8217' --header 'Authorization: Basic S0FTS05SVzU1RVVQWDJGVDhCUksyRzlEOmZsZDgxc05ZRHNuVGNUclVwRTR4ZkpWLXUtZUlzTFpKZnVaa2RucWY=' --form 'file=@"%s"' '''
make_meta = '''curl --location --request POST 'https://metadata-api.klaytnapi.com/v1/metadata' --header 'x-chain-id: 8217' --header 'Content-Type: application/json' --header 'Authorization: Basic S0FTS05SVzU1RVVQWDJGVDhCUksyRzlEOmZsZDgxc05ZRHNuVGNUclVwRTR4ZkpWLXUtZUlzTFpKZnVaa2RucWY=' --data '{"metadata": { "name": "%s", "description": "This is a flexme NFT Token", "image": "%s","data" : %s} ,"filename": "%s" }' '''
make_token = '''curl --location --request POST 'https://kip17-api.klaytnapi.com/v2/contract/flexmetoken/token' \
  --header "x-chain-id: 8217" \
  -u KASKNRW55EUPX2FT8BRK2G9D:fld81sNYDsnTcTrUpE4xfJV-u-eIsLZJfuZkdnqf \
	--data-raw '{
	  "to": "%s",
	  "id": "%s",
	  "uri": "%s"
	}'
'''


async def Image_upload(image):     
    try:
        test =  os.popen(image_uploader %(image)).read()
        json_data = json.loads(test.split('%')[0])
        
    except:
        json_data = {'code':1, 'msg':'image upload ERROR'}
    
    

    return json_data

async def Make_meta(result,data):
  
    try:
        image = result['uri']
        name = data['name']                                                                            #name
        file_name = name + '.json'                                                                      #filename
        metadata = str(data['metadata']).replace("'",'"')                                              #metadata
        #print(metadata)
        time.sleep(1) #seconds                                                                         #이미지가 서버에 저장되는 시간을 기다려 줘야함
        #print(make_meta % (name,image,metadata,file_name))
        json_data = os.popen(make_meta % (name,image,metadata,file_name)).read()
        #print(json_data)#
        err = (json.loads(json_data.split('%')[0])['filename'])      #error확인을 위한 호출
       
    except:
        json_data = {'code':1, 'msg':'Make jsonfile Error'}
    

    return json_data
    

async def Make_token(result,data):

    try:
        
        id = format(int(data['id']),"#x")                                    # 토큰 id
        wallet_address = data['wallet']                                      # wallet address 

        json_data = json.loads(result.split('%')[0])
   
        uri = json_data['uri']
        #print(make_token%(wallet_address,id,uri))
        result = os.popen(make_token%(wallet_address,id,uri)).read()
        #print(result)
        err = (json.loads(result.split('%')[0])['status'])                   #에러결과 걸러내기위한 출력
        

        
    except:
        result = {'code':1, 'msg':'Make_token ERROR'}

    return result


async def main(image,data):

    
   
    try:
        data = json.loads(data)

        #print('1')
        result = await Image_upload(image)
        uri = result['uri']
        data["image"]=uri

        #print('2')
        result = await Make_meta(result,data)
        json_data = json.loads(result.split('%')[0])
        uri = json_data['uri']
        data['uri']=uri

        #print('3')
        result = await Make_token(result,data)
        data.update(json.loads(result.split('%')[0]))
        result = json.loads(result)

        if(result['status']=='Submitted'):
        
            err = NftInfoAdd(data,str(data['uid']))
            
            if(err== 0):
                data = {'code':0, 'msg':'success'}
            else:
                data = err
                
                

    except:
        data= result
        

    return data
