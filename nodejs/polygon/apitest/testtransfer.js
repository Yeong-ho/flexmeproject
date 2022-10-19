// Setup: npm install alchemy-sdk
// Github: https://github.com/alchemyplatform/alchemy-sdk-js
const { Alchemy, Network, Wallet, Utils } =require("alchemy-sdk");
const dotenv = require("dotenv");

dotenv.config();
const { API_KEY, PRIVATE_KEY } = process.env;

const settings = {
  apiKey: API_KEY,
  network: Network.MATIC_MUMBAI, // Replace with your network.
};
const alchemy = new Alchemy(settings);

let wallet = new Wallet(PRIVATE_KEY);




async function main(){

    try{
const nonce = await alchemy.core.getTransactionCount(wallet.address, "latest");
let exampleTx = {
    to: "0xA47B14ed739433b968848F438C612FDd0bb1A950",
    value: '10000000000000000',
    gasLimit: "21000",
    maxPriorityFeePerGas: "5000000000",
    maxFeePerGas: "20000000000",
    nonce: nonce,
    type: 2,
    chainId: 80001,
  };

let rawTransaction = await wallet.signTransaction(exampleTx);



const signedTx = await alchemy.transact.sendTransaction(rawTransaction);

console.log(signedTx);

}
catch(err){
  console.log(err)
}

}


main();