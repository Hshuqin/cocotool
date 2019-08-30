import numpy as np
import os
import json
from PIL import Image
ch_file_path = '../mmdetection/data/crowdhuman/annotation_val.odgt'
#json_file_path = '../mmdetection/data/crowdhuman/crowdhuman2coco_val.json' 
json_file_path = '../mmdetection/data/crowdhuman/coco_val.json'
def load_func(fpath):#fpath是具体的文件 ，作用：#str to list
    assert os.path.exists(fpath)  #assert() raise-if-not
    with open(fpath,'r') as fid:
        lines = fid.readlines()
    records = [json.loads(line.strip('\n')) for line in lines] #str to list
    return records
def crowdhuman2coco(odgt_path,json_path):
    records = load_func(odgt_path)   #提取odgt文件数据
    
    json_dict = {"images":[], "annotations": [], "categories": []}#定义一个字典，coco数据集标注格
    START_B_BOX_ID = 1 #设定框的起始ID
    image_id = 1  #每个image的ID唯一，自己设定start，每
    bbox_id = START_B_BOX_ID
    image = {} #定义一个字典，记录image
    annotation = {} #记录annotation
    categories = {}  #进行类别记录
    record_list = len(records)  #获得record的长度，循环遍历所有数据
    print(record_list)
    #一行一行的处理
    for i in range(3):
        file_name = records[i]['ID']+'.jpg'  #这里是字符串格式  eg.273278,600e5000db6370fb
        #image_id = int(records[i]['ID'].split(",")[0]) 这样会导致id唯一，要自己设定
        im = Image.open("../mmdetection/data/crowdhuman/Images/"+file_name)
        image = {'file_name': file_name, 'height': im.height, 'width': im.width,'id':image_id} #im.size[0]，im.size[1]分别是宽
        
        json_dict['images'].append(image) #这一步完成一行数据到字典images的转换

        gt_box = records[i]['gtboxes']  
        gt_box_len = len(gt_box) #每一个字典gtboxes里，也有好几个记录，分别提取记录
        for j in range(gt_box_len):
            category = gt_box[j]['tag']
            if category not in categories:  #该类型不在categories，就添加上去
                new_id = len(categories) + 1 #ID递增
                categories[category] = new_id
            category_id = categories[category]  #重新获取它的类别ID
            fbox = gt_box[j]['fbox']  #获得全身
            #对ignore进行处理，ignore有时在key：extra里，有时在key：head_attr里。属于互斥的
            ignore = 0 #下面key中都没有ignore时，就设
            if "ignore" in gt_box[j]['head_attr']:
                ignore = gt_box[j]['head_attr']['ignore']
            if "ignore" in gt_box[j]['extra']:
                ignore = gt_box[j]['extra']['ignore']
            #对字典annotation进行设值
            annotation = {'area': fbox[2]*fbox[3], 'iscrowd': ignore, 'image_id':  #添加hbox、vbox字段
                        image_id, 'bbox':fbox, 'hbox':gt_box[j]['hbox'],'vbox':gt_box[j]['vbox'],
                        'category_id': category_id,'id': bbox_id,'ignore': ignore,'segmentation': []}  
            #area的值，暂且就是fbox的宽高相乘了，观察里面的数据，发现fbox[2]小、fbox[3]很大，刚好一个全身框的宽很小，高就很大。（猜测�?
            #segmentation怎么处理？这里就暂且不处理
            #hbox、vbox、ignore是添加上去的，以防有需要
            json_dict['annotations'].append(annotation)
            
            bbox_id += 1 #框ID ++
        image_id += 1
        #annotations的转化结束
    #下面这一步，对所有数据，只需执行一次
    for cate, cid in categories.items():
            #dict.items()返回列表list的所有列表项，形如这样的二元组list：［(key,value),(key,value),...
            cat = {'supercategory': 'none', 'id': cid, 'name': cate}
            json_dict['categories'].append(cat)
    #到此，json_dict的转化全部完成，对于其他的key
    #因为没有用到（不访问），就不需要给他们空间，也不需要去处理，字典是按key访问的
    json_fp = open(json_path, 'w')
    json_str = json.dumps(json_dict) #写json文件
    json_fp.write(json_str)
    json_fp.close()

#ch_file_path = '../mmdetection/data/crowdhuman/annotation_val.odgt'
#json_file_path = 'coco1.json' 
crowdhuman2coco(ch_file_path,json_file_path)



