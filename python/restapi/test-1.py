


test = [('data[name]', 'flexme_NFT_token59'), ('data[id]', '0x3b'), ('data[wallet]', '0xF84ae1Bdd0a7b18d71c470C4d032d8E68bb32Cb2'),('data[metadata][help]', '010-2222-3333'), ('data[metadata][time]', '2022'), ('data[metadata][makein]', 'developer')]


array = ['data[name]', 'data[id]', 'data[wallet]', 'data[metadata][help]', 'data[metadata][time]', 'data[metadata][makein]']

metadata = ""
meta_size = len(array)-3
print(meta_size)
for key in array:
    
    
    if key.split('[')[1].split(']')[0] == 'metadata':
            metadata=metadata+'"%s"'%key.split('[')[2].split(']')[0]+':'+'"%s"'%array[meta_size]+','
            meta_size+=1
metadata = metadata[:-1]
print(metadata)


