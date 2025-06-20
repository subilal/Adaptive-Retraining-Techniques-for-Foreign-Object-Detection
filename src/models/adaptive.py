from ultralytics import YOLO
from ultralytics import settings
import time
import os
os.environ['WANDB_DISABLED'] = 'true'

import matplotlib
matplotlib.use('Agg')

import shutil
import yaml
import os
from random import choice
import pandas as pd
import pickle
import numpy as np


num_splits = 10

Map_50_all_splits = []
Map_90_all_splits = []
R_all_splits= []
P_all_splits = []

class_acc__all_splits= []
limits__all_splits= []

samples_count_all_splits = []
num_info_samples_all_splits = []
num_adver_samples_all_splits = []

for Split_index in range(num_splits):
    dataset_yaml =  r"dataset_" + str(Split_index) + ".yaml"
    project = "Output/Adaptive_20_" + str(Split_index)
    batch = 32
    epochs = 100
    image_size = 1024
    device = 1
    single_cls = False
    
    if Split_index > 8 :
      epochs = 150
      

    model = YOLO('yolov8s.pt')
    result = model.train(data=dataset_yaml, project=project, batch=batch, epochs=epochs, verbose=False, workers=28, single_cls = False,imgsz =  image_size, device = 1 )
    
    project2 = project + "/Validation_2"
    results =  model.val(data = "dataset_Validation_2.yaml", project=project2, single_cls = False,imgsz =  image_size, device = 1 , save = True, save_json = True)
    # Perform validation
    
    data = {
    'class': [],
    'P': [],
    'R': [],
    'mAP50': [],
    'mAP50-95': [] }

    for class_indx in results.ap_class_index:
        out = results.class_result(class_indx)
        data['class'].append(class_indx)
        data['P'].append(out[0])
        data['R'].append(out[1])
        data['mAP50'].append(out[2])
        data['mAP50-95'].append(out[3])

    
    
    if Split_index > 7 :
      data['class'].append(100)
    else:
      data['class'].append(150)
    data['P'].append(results.results_dict['metrics/precision(B)'])
    data['R'].append(results.results_dict['metrics/recall(B)'])
    data['mAP50'].append(results.results_dict['metrics/mAP50(B)'])
    data['mAP50-95'].append(results.results_dict['metrics/mAP50-95(B)'])

    Map_50_all_splits.append(results.results_dict['metrics/mAP50(B)'])
    Map_90_all_splits.append(results.results_dict['metrics/mAP50-95(B)'])
    R_all_splits.append(results.results_dict['metrics/recall(B)'])
    P_all_splits.append(results.results_dict['metrics/precision(B)'])

    df = pd.DataFrame(data)
    df.to_csv(project2+ '/out.csv', index=False) 

    if Split_index == 9:
        break
    
    vald_pTH = str(results.save_dir) + "/Class_accuracy.txt"
    
    with open(vald_pTH, 'r') as fopn:
        acc = fopn.readlines()
    
    out_acc = acc[0].split()
    out_acc.remove(out_acc[0])
    out_acc.remove(out_acc[-1])
        
    for index in range(len(out_acc)):
        out_acc[index] = round(float(out_acc[index]),2)
        if np.isnan(out_acc[index]):
            out_acc[index] = 0

    limits = []
    
    upper_bound = 0.90
    confident_range = 0.20
    
    for x in out_acc:
        upper_limit = round(upper_bound + (1.0 - x) / 2,2)
        lower_limit = round(upper_limit - confident_range, 2)
        
        if upper_limit > 1.0 :
            upper_limit = 1.0
        if lower_limit > 1.0 :
            lower_limit = 1.0
        limits.append([upper_limit, lower_limit])

    class_acc__all_splits.append(out_acc)
    limits__all_splits.append(limits)

    image_path = r"/data/P70080563/IEEE Paper/Train_test_Split_All/DATA/" +  str(Split_index + 1) + "/images"
    pred_res = model.predict(image_path, conf = 0.1, save= True,save_conf = True,save_txt = True, save_json = True)
    labl_path = str(pred_res[0].save_dir) + '/labels'
    filenames = os.listdir(labl_path)
    
    dataframe = []
    dataframe2 = []
    
    for file in filenames:
        split_type = []
        filepath = labl_path + '/' + file
        
        with open(filepath, 'r' ) as fpo:
            lines = fpo.readlines()
            #print(lines)
            for line in lines:
                splits_array = line.split()
                class_category = int(splits_array[0])
                conf = round(float(splits_array[-1]),2)
                #print(class_category, conf)
                
                #print(class_category)
                if conf > limits[class_category][0]:
                    split = "Confident"
                elif conf <  limits[class_category][1]:
                    split = "Adversial"
                else:
                    split = "Informative"
               
                split_type.append(split)                
                #print(file, class_category, conf,split)
                dataframe.append([file, class_category,conf,  split ])
        
        if "Adversial" in split_type:
            final_type = "Adversial"
        elif "Informative" in split_type:
            final_type = "Informative"
        else:
            final_type = "Confident"
            
        dataframe2.append([file , final_type])     

    for index in dataframe:
        if index[3]=='Confident':
            dataframe.remove(index)

    sort1 = []
    for index in dataframe:
        sort1.append(index[1])

    from collections import Counter
    sort_det = Counter(sort1)
    pd_df = pd.DataFrame(dataframe, columns= ["filename","class","conf","split"])
    x = pd_df.groupby('class')
    x.get_group(0).sort_values(by ="conf" )
    pd_df2 = pd.DataFrame(columns= ["filename","class","conf","split"])
    
    
    list_cnm =x.describe().index.tolist()

    for indx in list_cnm:
        y = x.get_group(indx).sort_values(by ="conf" ).head(7)
        #print(y)
        pd_df2 = pd.concat([pd_df2, y ], ignore_index=True )

    Temp_data_path = "Temp_data_" + str(Split_index + 1)
    
    if os.path.exists(Temp_data_path):
        shutil.rmtree(Temp_data_path)
    
    os.makedirs(Temp_data_path, exist_ok = True)
    path1 = Temp_data_path
    
    dataset_new_adver = str(path1) + r"/adver/images"
    dataset_new_inform = str(path1) + r"/inform/images"
    os.makedirs(dataset_new_adver, exist_ok=True)
    os.makedirs(dataset_new_inform, exist_ok=True)
    
    label_predictions = labl_path
    label_ground = image_path.replace("images", "labels")
    
    label_new_adver = str(path1) + r"/adver/labels"
    label_new_inform = str(path1) + r"/inform/labels"
    
    os.makedirs(label_new_adver, exist_ok=True)
    os.makedirs(label_new_inform, exist_ok=True)

    adver_count  = 0
    info_count  = 0
    
    for dff in pd_df2.values:
        
        old_img_name = image_path + '/' + dff[0].replace(".txt",".bmp")
    
        if dff[3] == 'Adversial':
            new_img_name = dataset_new_adver + '/' + dff[0].replace(".txt",".bmp")
            shutil.copy(old_img_name,new_img_name)
            old_label_name = label_ground + '/' + dff[0]
            
            label_new_name = label_new_adver + '/' + dff[0]
            shutil.copy(old_label_name, label_new_name)
            adver_count += 1 
            
        elif dff[3] == 'Informative':
            info_count += 1
            new_img_name = dataset_new_inform + '/' + dff[0].replace(".txt",".bmp")
            shutil.copy(old_img_name,new_img_name)
            old_label_name = label_predictions + '/' + dff[0]
            
            label_new_name = label_new_inform + '/' + dff[0]
            shutil.copy(old_label_name, label_new_name)
            with open(label_new_name, 'r+') as fopen:
                lines = fopen.readlines()
                for inx in range(len(lines)):
                    lines[inx] = " ".join(lines[inx].split()[0:5]) + "\n"
                fopen.seek(0)
                fopen.truncate()
                fopen.writelines(lines)
                fopen.close()


    print("Adversial files count :  " , adver_count)
    print("Informative files count :  " , info_count)
    num_info_samples_all_splits.append(info_count)
    num_adver_samples_all_splits.append(adver_count)
    
    
    new_dataset_path = "Temp_data_All_" + str(Split_index + 1)
    if os.path.exists(new_dataset_path):
        shutil.rmtree(new_dataset_path)
        
    os.makedirs(new_dataset_path, exist_ok = True)
    
    shutil.copytree(dataset_new_adver, new_dataset_path,  dirs_exist_ok=True)
    shutil.copytree(dataset_new_inform, new_dataset_path,  dirs_exist_ok=True)
    shutil.copytree(label_new_adver, new_dataset_path,  dirs_exist_ok=True)
    shutil.copytree(label_new_inform, new_dataset_path,  dirs_exist_ok=True)

    if Split_index == 0:
        Split_0_images  = r"/data/P70080563/IEEE Paper/Train_test_Split_All/DATA/0/images/"
        Split_0_labels  = r"/data/P70080563/IEEE Paper/Train_test_Split_All/DATA/0/labels/"
        shutil.copytree(Split_0_images, new_dataset_path,  dirs_exist_ok=True)
        shutil.copytree(Split_0_labels, new_dataset_path,  dirs_exist_ok=True)
    else:
        Old_data = "Temp_data_All_" + str(Split_index)
        shutil.copytree(Old_data, new_dataset_path,  dirs_exist_ok=True)
    
    crsPath = new_dataset_path #dir where images and annotations stored
    output_dataset_paTH = r"dataset_"+str(Split_index + 1 ) # dir to save  new data
    
    if os.path.exists(output_dataset_paTH):
        shutil.rmtree(output_dataset_paTH)
    os.makedirs(output_dataset_paTH, exist_ok = True)

    path1 = output_dataset_paTH
    #arrays to store file names
    imgs =[]
    xmls =[]
    #setup ratio (val ratio = rest of the files in origin dir after splitting into train and test)
    train_ratio = 0.8
    val_ratio = 0.2
    
    #total count of imgs
    totalImgCount = len(os.listdir(crsPath))/2
    
    #soring files to corresponding arrays
    for (dirname, dirs, files) in os.walk(crsPath):
        for filename in files:
            if filename.endswith('.txt'):
                xmls.append(filename)
            else:
                imgs.append(filename)

    #counting range for cycles
    countForTrain = int(len(imgs)*train_ratio)
    countForVal = int(len(imgs)*val_ratio)
    print("training images are : ",countForTrain)
    print("Validation images are : ",countForVal)

    samples_count_all_splits.append([countForTrain, countForVal])
    
    trainimagePath = output_dataset_paTH + r'/images/train'
    trainlabelPath = output_dataset_paTH + r'/labels/train'
    valimagePath = output_dataset_paTH + r'/images/val'
    vallabelPath = output_dataset_paTH + r'/labels/val'
    
    os.makedirs(trainimagePath, exist_ok=True)
    os.makedirs(trainlabelPath, exist_ok=True)
    os.makedirs(valimagePath, exist_ok=True)
    os.makedirs(vallabelPath, exist_ok=True)
    
    #cycle for train dir
    for x in range(countForTrain):
    
        fileJpg = choice(imgs) # get name of random image from origin dir
        fileXml = fileJpg[:-4] +'.txt' # get name of corresponding annotation file
    
        #move both files into train dir
        #shutil.move(os.path.join(crsPath, fileJpg), os.path.join(trainimagePath, fileJpg))
        #shutil.move(os.path.join(crsPath, fileXml), os.path.join(trainlabelPath, fileXml))
        shutil.copy(os.path.join(crsPath, fileJpg), os.path.join(trainimagePath, fileJpg))
        shutil.copy(os.path.join(crsPath, fileXml), os.path.join(trainlabelPath, fileXml))
    
    
        #remove files from arrays
        imgs.remove(fileJpg)
        xmls.remove(fileXml)
    
    #cycle for test dir   
    for x in range(countForVal):
    
        fileJpg = choice(imgs) # get name of random image from origin dir
        fileXml = fileJpg[:-4] +'.txt' # get name of corresponding annotation file
    
        #move both files into train dir
        #shutil.move(os.path.join(crsPath, fileJpg), os.path.join(valimagePath, fileJpg))
        #shutil.move(os.path.join(crsPath, fileXml), os.path.join(vallabelPath, fileXml))
        shutil.copy(os.path.join(crsPath, fileJpg), os.path.join(valimagePath, fileJpg))
        shutil.copy(os.path.join(crsPath, fileXml), os.path.join(vallabelPath, fileXml))
        
        #remove files from arrays
        imgs.remove(fileJpg)
        xmls.remove(fileXml)
    
    #rest of files will be validation files, so rename origin dir to val dir
    #os.rename(crsPath, valPath)
    #shutil.move(crsPath, valPath) 
    dataset_yaml_data = {
             'train': trainimagePath,
             'val': valimagePath,
             'nc': 16,
             'names': [
        'cells',
        'Drill_Bits',
        'Nuts',
        'Paper',
        'Plastic_Part',
        'Plastic_Strips',
        'Rings',
        'Screws',
        'Square_Metal',
        'T_Tool',
        'Wire_Cuts',
        'Stopper',
        'U_Pin',
        'Screw_Driver',
        'Plier',
        'Spanner'
    ]}  
    
    split_yaml_name = "dataset_" + str(Split_index +1) + ".yaml"
    
    with open(split_yaml_name, "w") as fp:
        yaml.dump(dataset_yaml_data, fp, default_flow_style=False)
    print("Ending one iteration")





filehandler = open("Map_50_all_splits.obj","wb")
pickle.dump(Map_50_all_splits,filehandler)
filehandler.close()

filehandler = open("Map_90_all_splits.obj","wb")
pickle.dump(Map_90_all_splits,filehandler)
filehandler.close()

filehandler = open("R_all_splits.obj","wb")
pickle.dump(R_all_splits,filehandler)
filehandler.close()

filehandler = open("P_all_splits.obj","wb")
pickle.dump(P_all_splits,filehandler)
filehandler.close()

filehandler = open("class_acc__all_splits.obj","wb")
pickle.dump(class_acc__all_splits,filehandler)
filehandler.close()

filehandler = open("limits__all_splits.obj","wb")
pickle.dump(limits__all_splits,filehandler)
filehandler.close()

filehandler = open("samples_count_all_splits.obj","wb")
pickle.dump(samples_count_all_splits,filehandler)
filehandler.close()

filehandler = open("num_info_samples_all_splits.obj","wb")
pickle.dump(num_info_samples_all_splits,filehandler)
filehandler.close()

filehandler = open("num_adver_samples_all_splits.obj","wb")
pickle.dump(num_adver_samples_all_splits, filehandler)
filehandler.close()
