var express = require('express');
var router = express.Router();
const { Network, Alchemy } = require('alchemy-sdk');

const settings = {
   apiKey: 'uYzjvFceiP2QsnX9lJRaix8GWuC5OCTJ', // Replace with your Alchemy API Key.
   network: Network.MATIC_MAINNET, // Replace with your network.
};

const alchemy = new Alchemy(settings);

/* GET users listing. */
router.get('/:wallet', async (req, res, next) => {

  try{
  let wallet = req.params.wallet;

  nftlist = await getNfts(wallet);
  }
  catch(err){res.send(err);}
  
  res.send(nftlist);
});

module.exports = router;


async function getNfts(wallet){

  let data = [];
  nftlist = await alchemy.nft.getNftsForOwner(wallet);
  
  for(idx in nftlist.ownedNfts){
    
   let tmp = {
      contract:nftlist.ownedNfts[idx].contract.address, 
      tokenId: nftlist.ownedNfts[idx].tokenId,

    };
    data.push(tmp);
  };
  



return data;
}