import os
import pandas  as pd


with open('/Users/yh/Desktop/C-VTON/data/viton/viton_train_pairs.txt',"r") as df:
    test = df.read()
f =open('./example.txt',"w")
dir_path = "/Users/yh/Desktop/mmfashion/data/VTON/viton_resize/image/image"
close_path = "/Users/yh/Desktop/mmfashion/data/VTON/viton_resize/image/cloth"

t = test.split("\n")
result =[]
result_1 = []


for (root, directories, files) in os.walk(dir_path):
    for file in files:
        file_path = os.path.join(root, file)
        for i in t:
            if file_path.split("/")[10]==i.split(" ")[0]:
                result.append(i.split(" ")[0].split("_")[0]+"_1.jpg")
            '''
            if file_path.split("/")[10]==i.split(" ")[0]:
                tmp = i.split(" ")[0].split("_")[0]+"_1.jpg"
                f.write("%s %s \n" % (i.split(" ")[0],tmp))
            '''
        
        #print(file_path.split("/")[10])

print(len(result))


for (root, directories, files) in os.walk(close_path):
    for file in files:
        file_path = os.path.join(root, file)
        for i in result:
            if file == i:
                tmp = i.split(" ")[0].split("_")[0]+"_0.jpg"
                f.write("%s %s\n" % (tmp ,i))











df.close()
f.close()

