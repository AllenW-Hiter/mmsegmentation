import os
import os.path as osp 
import cv2 as cv
from tqdm import tqdm
from typing import Optional,Tuple

def label_normal(input_root:str,output_root:Optional[str]=None,suffix:Optional[str]=None,resize:Optional[Tuple[int,int]]=None)->None:
    if not output_root:
        output_root=input_root
    if not osp.exists(output_root):
        os.makedirs(output_root)
    files_names=os.listdir(input_root)
    for files_name in tqdm(files_names,total=len(files_names)):
        name=files_name.split(".")[0]
        img=cv.imread(osp.join(input_root,files_name),cv.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error reading {files_name}. Skipping this file.")
            continue
        if resize:
            img = cv.resize(img, resize, interpolation=cv.INTER_NEAREST)
        img[img==0]=255
        img[img!=255]=0
        out_file_name = name +".png" if  not suffix else name + suffix +".png"
        cv.imwrite(osp.join(output_root,out_file_name),img=img)


if __name__=="__main__":
    mask_input_root="/home/wxy/02_MyProjects/MMLab_Baseline/Dataset/DLOsSS/annotations/val_or"
    mask_output_root="/home/wxy/02_MyProjects/MMLab_Baseline/Dataset/DLOsSS/annotations/val"
    suffix="_label"
    resize=(1024,1024)
    label_normal(mask_input_root,mask_output_root,suffix)
    
    print("test")