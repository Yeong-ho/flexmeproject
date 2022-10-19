var express = require('express');
var router = express.Router();
const { Network, Alchemy } = require('alchemy-sdk');

const settings = {
   apiKey: 'uYzjvFceiP2QsnX9lJRaix8GWuC5OCTJ', // Replace with your Alchemy API Key.
   network: Network.MATIC_MUMBAI, // Replace with your network.
};

const alchemy = new Alchemy(settings);

/* GET users listing. */
router.get('/:contract/:tokenid', async (req, res, next) => {

  try{
  let contract = req.params.contract;
  let tokenid = req.params.tokenid;
  console.log(contract+'\n'+tokenid);
  nftlist = await getOwners(contract,tokenid);
  }
  catch(err){res.send(err);}
  
  res.send(nftlist);
});

module.exports = router;


async function getOwners(contract,tokenid){

 
 let data = await alchemy.nft.getOwnersForNft(contract,tokenid);
  


return data;
}