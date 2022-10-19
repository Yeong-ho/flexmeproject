const { Network, Alchemy } = require('alchemy-sdk');


// Optional Config object, but defaults to demo api-key and eth-mainnet.
const settings = {
  apiKey: 'uYzjvFceiP2QsnX9lJRaix8GWuC5OCTJ', // Replace with your Alchemy API Key.
  network: Network.MATIC_MAINNET, // Replace with your network.

};

const alchemy = new Alchemy(settings);

// Print all NFTs returned in the response:
alchemy.nft.getNftsForOwner("0xDf7A6c5a993E8E0022A3c94F9990E896d6952e8e").then(console.log);
