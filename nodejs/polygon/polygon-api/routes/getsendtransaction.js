var express = require('express');
var router = express.Router();
const Web3 = require('web3');
const { Network, Alchemy,Wallet } = require('alchemy-sdk');

const settings = {
   apiKey: 'uYzjvFceiP2QsnX9lJRaix8GWuC5OCTJ', // Replace with your Alchemy API Key.
   network: Network.MATIC_MUMBAI, // Replace with your network.
};

const alchemy = new Alchemy(settings);

/* GET users listing. */
router.get('/', async (req, res, next) => {

try{
    
    let toaddress = req.body.to;
    let wallet = new Wallet(req.body.from);
    let value = req.body.value;
    
    const nonce = await alchemy.core.getTransactionCount(wallet.address, "latest");
    let exampleTx = {
    to: toaddress,
    value: Web3.utils.toWei(value,"ether"),
    gasLimit: "21000",
    maxPriorityFeePerGas: Web3.utils.toWei("5","Gwei"),
    maxFeePerGas: Web3.utils.toWei("20","Gwei"),
    nonce: nonce,
    type: 2,
    chainId: 80001,
  };

let rawTransaction = await wallet.signTransaction(exampleTx);



var signedTx = await alchemy.transact.sendTransaction(rawTransaction);

console.log(signedTx);

}
    catch(err){res.send(err);}


  
res.send(signedTx);
});

module.exports = router;



