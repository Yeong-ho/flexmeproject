require('dotenv').config();
const { createAlchemyWeb3 } = require('@alch/alchemy-web3');
const { isValidAddress } = require('ethereumjs-util');

const contractABI = require('./contractABI');

const startAtTokenId = 0; // Modify this value depending on the NFT Token ID you want to transfer
const NFT_CONTRACT_ADDRESS = ''; // Replace with the contract address of the NFTs 
const ADDRESS_LIST = []; // Destination addresses where the NFTs will be transferred

const { API_URL, PUBLIC_KEY, PRIVATE_KEY } = process.env;
const web3 = createAlchemyWeb3(API_URL);
let initialNonce = null;

const sendSignedTransaction = (signedTx) =>
  new Promise((resolve, reject) => {
    web3.eth.sendSignedTransaction(signedTx.rawTransaction, (err, hash) => {
      if (!err) {
        console.log(
          'The hash of your transaction is: ',
          hash,
          "\nCheck Alchemy's Mempool to view the status of your transaction!",
        );
        return resolve(hash);
      }

      console.log(
        'Something went wrong when submitting your transaction:',
        err,
      );
      return reject(err);
    });
  });

const transferNFT = async (toAddress, tokenId, index) => {
  const nftContract = new web3.eth.Contract(contractABI, NFT_CONTRACT_ADDRESS);
  
  if (!initialNonce) {
    initialNonce = await web3.eth.getTransactionCount(PUBLIC_KEY, 'latest'); // get latest nonce
  }
  
  const nonce = initialNonce + index;

  const tx = {
    from: PUBLIC_KEY,
    to: NFT_CONTRACT_ADDRESS,
    gas: 500000,
    nonce,
    maxPriorityFeePerGas: 2999999987,
    value: 0,
    data: nftContract.methods
      .transferFrom(PUBLIC_KEY, toAddress, tokenId)
      .encodeABI(),
  };

  const signedTx = await web3.eth.accounts.signTransaction(tx, PRIVATE_KEY);

  const hash = await sendSignedTransaction(signedTx);
  return hash;
};

const main = async () => {
  if (!ADDRESS_LIST || !ADDRESS_LIST.length || !Array.isArray(ADDRESS_LIST)) {
    console.log('Invalid ADDRESS_LIST, must be an array');
    process.exit(1);
  }

  let hasInvalidAddresses = false;
  const invalidAddressList = [];

  ADDRESS_LIST.forEach((address) => {
    if (!address || !isValidAddress(address)) {
      hasInvalidAddresses = true;
      invalidAddressList.push(address);
    }
  });

  if (hasInvalidAddresses) {
    console.log('Has invalid addresses =>', invalidAddressList);
    process.exit(1);
  }

  const result = [];

  try {
    for (let index = 0; index < ADDRESS_LIST.length; index++) {
      try {
        const address = ADDRESS_LIST[index];
        const tokenId = index + startAtTokenId; 

        const transactionID = await transferNFT(address, tokenId, index);
        const time = new Date().toLocaleString();

        result.push({
          address,
          time,
          transactionID,
          tokenId,
        });
        console.log(result);
      } catch (error) {
        console.log('Error inside the for loop:', error);
      }
    }
  } catch (error) {
    console.log('Error:', error);
    process.exit(1);
  }
};

main();