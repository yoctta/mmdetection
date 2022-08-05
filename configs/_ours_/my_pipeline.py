from mmdet.datasets.builder import PIPELINES
import json 
import cv2
import numpy as np
import os
from random import sample

with open('/mnt/data/LVIS/id_map.json') as f:
    id_map_f=json.load(f)
label2cat = {id_map_f[cat]-1:cat for  cat in id_map_f}

def intersection(s1,s2):
    area2=(s2[3]-s2[1])*(s2[2]-s2[0])
    dx = min(s2[3], s1[3]) - max(s2[1], s1[1])
    dy = min(s2[2], s1[2]) - max(s2[0], s1[0])
    if (dx>=0) and (dy>=0):
        return dx*dy/area2
    else:
        return 0

@PIPELINES.register_module()
class In_N_Out:
    def __init__(self,subs_file='/mnt/home/syn4det/LVIS_l20_328_s28_subs.json',P=0.5,N=1,scale_p=[0.2,2],care_overlap=True,mask_threshold=128,care_cat=True):
        self.care_cat=care_cat
        with open(subs_file) as f:
            self.subs_dict=json.load(f)        
        self.P=P
        self.N=N
        self.scale_p=scale_p
        self.care_overlap=care_overlap
        self.mask_threshold=mask_threshold

    def load_RGBA_BB(self,file_path,size):
        img_RGBA=cv2.imread(file_path,cv2.IMREAD_UNCHANGED)
        alpha=img_RGBA[...,3:]
        RGB=img_RGBA[...,:3]
        seg_mask = np.where(alpha>self.mask_threshold)
        y_min,y_max,x_min,x_max = np.min(seg_mask[0]), np.max(seg_mask[0]), np.min(seg_mask[1]), np.max(seg_mask[1])
        scale=size/max((y_max-y_min),(x_max-x_min))
        new_H=scale*(y_max-y_min)
        new_W=scale*(x_max-x_min)
        RGB=cv2.resize(RGB[y_min:y_max,x_min:x_max],(round(new_W),round(new_H)))
        alpha=cv2.resize(alpha[y_min:y_max,x_min:x_max],(round(new_W),round(new_H)))/255
        return RGB,alpha

    def try_add_syn(self,img,bboxes,labels,masks,cls,care_overlap):
        catego=label2cat[cls]
        img_h,img_w=img.shape[:2]
        if catego not in self.subs_dict or len(self.subs_dict[catego])==0:
            return 0
        sub_img=sample(self.subs_dict[catego],1)[0]
        scales=[]
        for lab,bbox in zip(labels,bboxes):
            if lab==cls:
                scales.append(max(bbox[3]-bbox[1],bbox[2]-bbox[0]))
        scale=np.mean(scales)*np.random.uniform(*self.scale_p)
        try:
            RGB,alpha=self.load_RGBA_BB(sub_img,float(scale))
        except:
            return 0
        ph,pw=RGB.shape[:2]
        dy=img_h-ph
        dx=img_w-pw
        if dy<=0 or dx<=0:
            return 0   
        dy=np.random.randint(dy)
        dx=np.random.randint(dx)
        if care_overlap:
            for bbox in bboxes:
                if intersection([dx,dy,dx+pw,dy+ph],bbox)>0.2:
                    return 0
        labels.append(cls)
        bboxes.append([dx,dy,dx+pw,dy+ph])
        _mask=np.zeros([img_h,img_w])
        _mask[dy:dy+ph,dx:dx+pw]=alpha
        masks.append((_mask>self.mask_threshold/255).astype('uint8'))
        img[dy:dy+ph,dx:dx+pw]=img[dy:dy+ph,dx:dx+pw]*(1-alpha[...,None])+RGB*alpha[...,None]
        return 1 

    def __call__(self,results):
        img=results['img']
        bboxes=results['gt_bboxes'].tolist()
        labels=results['gt_labels'].tolist()
        masks=[i for i in results['gt_masks'].masks]
        if self.care_cat:
            label_set=set(labels)
            assert len(labels)==len(bboxes)
            N = min(self.N,len(label_set))
            sample_labels=sample(label_set,N)
        else:
            sample_labels=sample(list(label2cat),N)
        for i in sample_labels:
            if np.random.rand()<=self.P:
                for _ in range(3):
                    if self.try_add_syn(img,bboxes,labels,masks,i,self.care_overlap):
                        break
        if len(bboxes)>0:
            results['gt_bboxes']=np.array(bboxes,dtype='float32')
        if len(labels)>0:
            results['gt_labels']=np.array(labels)
        if len(masks)>0:
            results['gt_masks'].masks=np.stack(masks,0)
        return results