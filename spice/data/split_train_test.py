import os
import random

def split(full_list,shuffle=False,ratio=0.2):
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total==0 or offset<1:
        return [],full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1,sublist_2


if __name__ == "__main__":
    dir_path="/git/segment_images"
    image_path_list=os.listdir(dir_path)
    image_list=[]
    for file_name in image_path_list:
        if file_name.endswith(".jpg"):
            image_list.append(file_name)

    dataset_train,dataset_test = split(image_list,shuffle=True,ratio=0.8)
    save_dir="/git/assets"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir,"kangqiang_dataset_train.txt"), 'w+') as f:
        for image_path  in dataset_train:
            f.write(image_path)
            f.write("\n")
        f.close()
    with open(os.path.join(save_dir,"kangqiang_dataset_test.txt"), 'w+') as f:
        for image_path in dataset_test:
            f.write(image_path)
            f.write("\n")
        f.close()

    

