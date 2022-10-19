const { Network, Alchemy } = require('alchemy-sdk');

// Optional Config object, but defaults to demo api-key and eth-mainnet.
const settings = {
  apiKey: 'demo', // Replace with your Alchemy API Key.
  network: Network.MATIC_MAINNET, // Replace with your network.
};

const alchemy = new Alchemy(settings);
alchemy.core.getBlockNumber().then(console.log);
