

import os

cmd = '''node -e "{const Caver = require('caver-js')\nconst caver = new Caver('https://api.baobab.klaytn.net:8651/')\nconst keyring = caver.wallet.keyring.generate()\nconsole.log(keyring) }"'''
walletKey = os.popen(cmd).read()

print(walletKey)
