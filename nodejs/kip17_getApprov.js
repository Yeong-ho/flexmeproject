const Caver = require('caver-js');
//const caver = new Caver('https://public-node-api.klaytnapi.com/v1/cypress');
const caver = new Caver('https://api.baobab.klaytn.net:8651');
const kip17 = new caver.kct.kip17('0x4beb2d4500523e710db2ec46bad4749c1cba0af2');
kip17.tokenURI(2).then(console.log);
//kip17.ownerOf(2).then(console.log);
//kip17.getApproved(2).then(console.log);
//kip17.isApprovedForAll('0xf84ae1bdd0a7b18d71c470c4d032d8e68bb32cb2','0x408F1F324E8516B28C6E066005Cd58ACa4789E3f').then(console.log);

