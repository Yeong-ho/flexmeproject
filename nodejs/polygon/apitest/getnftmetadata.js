const { Network, Alchemy } = require('alchemy-sdk');


// Optional Config object, but defaults to demo api-key and eth-mainnet.
const settings = {
  apiKey: 'uYzjvFceiP2QsnX9lJRaix8GWuC5OCTJ', // Replace with your Alchemy API Key.
  network: Network.MATIC_MAINNET, // Replace with your network.

};

const alchemy = new Alchemy(settings);

// Print all NFTs returned in the response:

alchemy.nft.getNftMetadata(
  "0xdc3a1d0db41d9cbc43abb61ee627d675f3730b2b",
  "621386449049956428958971126096683853"
).then(console.log);



