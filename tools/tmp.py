import os,re
file_path="/git/results/kangqiang"
fileList= os.listdir(file_path)

for filename in fileList:
    if filename.startswith("aug_plus"):
        aa=re.findall(r'[a-z]+|[0-9]+',filename)
        print(aa[2])
        newname=file_path+os.sep+"checkpoint_"+aa[2]+".pth.tar"
        print(newname)
        os.rename(filename, newname)