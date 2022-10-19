const CaverExtKAS = require('caver-js-ext-kas');
const caver = new CaverExtKAS(1001,"KASKXWR0OYXD889K1ZSPZUB6","ZOlDQ5FuSw9JVVqVseEKD_El7OS35U5-ipq2dAdH");

async function sendklay(){
try{

let contract = await caver.kas.kip17.getTokenList('my-first-kip17');

console.log(contract);
/*
let result = await caver.kas.kip17.transfer('my-first-kip17',"0x9D8aBbF3F0B8aFE4F1c38C5eD05e0Cf7ecDF4FaC","0x9D8aBbF3F0B8aFE4F1c38C5eD05e0Cf7ecDF4FaC","0xd490f897611635300bde23f5d51f0fc8c5ad0fa7",'0x2');
*/
}catch(err){console.log(err)}
}

sendklay();
