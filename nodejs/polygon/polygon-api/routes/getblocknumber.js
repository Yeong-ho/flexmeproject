var express = require('express');
var router = express.Router();
const { Network, Alchemy } = require('alchemy-sdk');

const settings = {
   apiKey: 'uYzjvFceiP2QsnX9lJRaix8GWuC5OCTJ', // Replace with your Alchemy API Key.
   network: Network.MATIC_MAINNET, // Replace with your network.
};

const alchemy = new Alchemy(settings);

/* GET users listing. */
router.get('/', async (req, res, next) => {

  num = req.query.num;
  console.log(num);
  blocknumber = await getBlockNumber(num);
  
  
  res.send(blocknumber);
});

module.exports = router;


async function getBlockNumber(num){

  blocknumber = await alchemy.core.getBlockNumber(num);
  
  data = blocknumber.toString();
return data;
}