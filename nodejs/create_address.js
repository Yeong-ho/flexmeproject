const Caver = require('caver-js')
const caver = new Caver('https://api.baobab.klaytn.net:8651/')

const keyring = caver.wallet.keyring.generate() // *
console.log(keyring)
