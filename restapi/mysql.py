from distutils.log import error
import os
import pymysql
import json


def SelectWallet(userId,userName):

    try:
        conn = pymysql.connect(host='ai-pri-admin-prod-test.cofzbians6ct.ap-northeast-2.rds.amazonaws.com',user='admin',password='12341234',db='NFTINFO',charset='utf8')
        cur = conn.cursor()
    
        select = cur.execute('select * from WALLET_INFO WHERE user_id = "%s" and user_name = "%s"'%(userId,userName))


        if str(select) == '0':                                                            #Wallet address가 없는 사용자
            select = cur.execute('select * from WALLET_INFO WHERE user_id = "%s"'%(userId))
            if str(select) =='0':
                cmd = '''node -e "{const Caver = require('caver-js')\nconst caver = new Caver('https://api.baobab.klaytn.net:8651/')\nconst keyring = caver.wallet.keyring.generate()\nconsole.log(keyring) }"'''
                walletKey = os.popen(cmd).read()          #wallet 생성
            
                address = walletKey.split("'")[1]
                pri_key = walletKey.split("'")[3]
                walletKey.close()
                select = cur.execute("INSERT INTO WALLET_INFO (user_id,user_name, address,Private_Key) VALUES(%s,'%s','%s','%s')"%(userId,userName,address,pri_key))
                conn.commit()
            
                result = address
          
                return result



            else :return 'Name Error'                                                  #해당ID  존재




        else:                                                                           #기존 Wallet address 사용자
            result = cur.fetchone()
            result = str(result[2])
    except: result = {"code":"1","msg":"select error"}
   
    #print(result)
    
    cur.close()
    return result

    
def NftInfoAdd(result,uid):
   
    try:

        conn = pymysql.connect(host='ai-pri-admin-prod-test.cofzbians6ct.ap-northeast-2.rds.amazonaws.com',user='admin',password='12341234',db='NFTINFO',charset='utf8')
        cur = conn.cursor()
        select = cur.execute("INSERT INTO NFT_INFO (token_id,nft_name, media_url,description) VALUES(%s,'%s','%s','%s')"%(result['id'],result['name'],result['image'],''))
        conn.commit()
        
        select = cur.execute("INSERT INTO NFT_TRANSACTION (user_id,token_id,meta_data,nft_transaction) VALUES(%s,%s,'%s','%s')"%(uid,result['id'],result['uri'],result['transactionHash']))
        conn.commit()
        conn.close()
        result = 0
    except: 
        
        result = {'code':1,'message':'db insert error'}

    return result
