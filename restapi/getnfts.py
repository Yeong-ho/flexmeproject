import os
import pymysql
import json
import requests


def getNftList(userId):
    conn = pymysql.connect(host='ai-pri-admin-prod-test.cofzbians6ct.ap-northeast-2.rds.amazonaws.com',user='admin',password='12341234',db='NFTINFO',charset='utf8')
    
    nftlist = {}
    nftlist['list']=[]
    

    headers = {"Content-Type": "application/json"}
    with conn:
        with conn.cursor() as cur:
            count =cur.execute('select token_id,meta_data,nft_transaction from NFT_TRANSACTION WHERE user_id = "%s" order by token_id desc;'%(userId))
            result = cur.fetchall()

            if count >0:
                for data in result:
                    response = requests.request("GET",data[1],headers=headers)
                    jsondata= json.loads(response.text)
                    metadata = {
                        "name": jsondata['name'],
                        "image": jsondata['image'],
                        "description":jsondata['description'],
                        "token_id": data[0],
                        "data": jsondata['data'],
                        "transactionhash": data[2]
                        }
                        #%(jsondata['name'],jsondata['image'],jsondata['description'],data[0],jsondata['data'])
                
                             
                    nftlist['list'].append(metadata)
            else : return {'code':1, 'msg':'No UID'}
   
    

    cur.close()

    return nftlist



def getNftMetadata(userid,tokenid):
    conn = pymysql.connect(host='ai-pri-admin-prod-test.cofzbians6ct.ap-northeast-2.rds.amazonaws.com',user='admin',password='12341234',db='NFTINFO',charset='utf8')
    
    
    
    

    headers = {"Content-Type": "application/json"}
    with conn:
        with conn.cursor() as cur:
            count = cur.execute('select token_id,meta_data,nft_transaction from NFT_TRANSACTION WHERE user_id = "%s" and token_id = "%s";'%(userid,tokenid))
            result = cur.fetchone()

            if count > 0:
                response = requests.request("GET",result[1],headers=headers)
                jsondata= json.loads(response.text)
                metadata = {
                        "name": jsondata['name'],
                        "image": jsondata['image'],
                        "description":jsondata['description'],
                        "token_id": tokenid,
                        "data": jsondata['data'],
                        "transactionhash": result[2]
                        }
            else : return {'code':1, 'msg':'UID no have token'}
    

    cur.close()

    return metadata