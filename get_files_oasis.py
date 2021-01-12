import os
import pandas as pd
import random
from random import shuffle
import numpy as np


def prepare_train_files(path):

    # get the file lists
    # Lists all the files present in path as a list
    pats = os.listdir(path)
    pat_path_list = []

    for pat in pats: 

        # Create a path with dir + patient folder
        pat_path = os.path.join(path, pat)
        pat_path_list.append(pat_path)

    # At the end of iteration, pat_path_list contains list of paths of all patients tf record files
    # rest data
    shuffle(pat_path_list)
    path_length = len(path)
    
    # Read the labels csv file (converted from dicom header to a csv)
    patientInfo = pd.read_csv('/home/d1353/no_backup/d1353/Code/OASIS_train.csv')
        
    '''
        Train NC patients
    '''
    label_lists = []
    id_lists = []
    cdr_lists = []
    
    y1 = patientInfo.loc[patientInfo['cdr'] == 0]
    y3 = list(y1['age'].unique())
    y3.sort()
    
    c = []
    for i in y3:
        c.append(y1.loc[y1['age'] == i,'id'].values.tolist())
        
    #print(c)
    print(len(c),len(y3))

    final_healthypatients = []
    for j in range(0,len(c)):
        if len(c[j]) < 2:
            final_healthypatients.append(c[j])
        else:
            final_healthypatients.append(random.sample(c[j], 2))

    final_healthypatients = [item for sublist in final_healthypatients for item in sublist]
    print(len(final_healthypatients))
    
    shuffle(final_healthypatients)
    
    tr_healthy = [path for path in pat_path_list if path[path_length:-9] in final_healthypatients]
    
    for id in final_healthypatients:    
        label_lists.append(y1.loc[y1['id'] == id,'age'].values[0])
        id_lists.append(y1.loc[y1['id'] == id,'id'].values[0])
        cdr_lists.append(y1.loc[y1['id'] == id,'cdr'].values[0])
        
    print(len(label_lists))
    print(type(label_lists[0]))
    
    print(len(id_lists))
    print(type(id_lists[0]))
    
    df = pd.DataFrame(data=label_lists)
    df['id'] = id_lists
    df['cdr'] = cdr_lists

    df.to_csv('/home/d1353/no_backup/d1353/Code/ResNet34/Shuffled_labels_trainhealthy.csv')
    
    '''
        Train AD patients
    '''
    label_lists = []
    id_lists = []
    cdr_lists = []
    
    y1 = patientInfo.loc[patientInfo['cdr'] != 0]
    y3 = list(y1['age'].unique())
    y3.sort()
    
    c = []
    for i in y3:
        c.append(y1.loc[y1['age'] == i,'id'].values.tolist())
        
    #print(c)
    print(len(c),len(y3))

    final_adpatients = []
    for j in range(0,len(c)):
        if len(c[j]) < 2:
            final_adpatients.append(c[j])
        else:
            final_adpatients.append(random.sample(c[j], 2))

    final_adpatients = [item for sublist in final_adpatients for item in sublist]
    print(len(final_adpatients))
    
    shuffle(final_adpatients)
    
    tr_ad = [path for path in pat_path_list if path[path_length:-9] in final_adpatients]
    
    for id in final_adpatients:    
        label_lists.append(y1.loc[y1['id'] == id,'age'].values[0])
        id_lists.append(y1.loc[y1['id'] == id,'id'].values[0])
        cdr_lists.append(y1.loc[y1['id'] == id,'cdr'].values[0])
        
    print(len(label_lists))
    print(type(label_lists[0]))
    
    print(len(id_lists))
    print(type(id_lists[0]))
    
    df = pd.DataFrame(data=label_lists)
    df['id'] = id_lists
    df['cdr'] = cdr_lists

    df.to_csv('/home/d1353/no_backup/d1353/Code/ResNet34/Shuffled_labels_trainAD.csv')
        
    return tr_healthy, tr_ad
    

def prepare_eval_files(path):

    # get the file lists
    # Lists all the files present in path as a list
    pats = os.listdir(path)
    pat_path_list = []

    for pat in pats: 

        # Create a path with dir + patient folder
        pat_path = os.path.join(path, pat)
        pat_path_list.append(pat_path)

    # At the end of iteration, pat_path_list contains list of paths of all patients tf record files
    # rest data
    shuffle(pat_path_list)
    path_length = len(path)
    
    # Read the labels csv file (converted from dicom header to a csv)
    patientInfo = pd.read_csv('/home/d1353/no_backup/d1353/Code/OASIS_test.csv')
        
    '''
        Eval NC patients
    '''
    label_lists = []
    id_lists = []
    cdr_lists = []
    
    y1 = patientInfo.loc[patientInfo['cdr'] == 0]
    y3 = list(y1['age'].unique())
    y3.sort()
    
    c = []
    for i in y3:
        c.append(y1.loc[y1['age'] == i,'id'].values.tolist())
        
    #print(c)
    print(len(c),len(y3))

    final_healthypatients = []
    for j in range(0,len(c)):
        if len(c[j]) < 2:
            final_healthypatients.append(c[j])
        else:
            final_healthypatients.append(random.sample(c[j], 2))

    final_healthypatients = [item for sublist in final_healthypatients for item in sublist]
    print(len(final_healthypatients))
    
    shuffle(final_healthypatients)
    
    eval_healthy = [path for path in pat_path_list if path[path_length:-9] in final_healthypatients]
    
    for id in final_healthypatients:    
        label_lists.append(y1.loc[y1['id'] == id,'age'].values[0])
        id_lists.append(y1.loc[y1['id'] == id,'id'].values[0])
        cdr_lists.append(y1.loc[y1['id'] == id,'cdr'].values[0])
        
    print(len(label_lists))
    print(type(label_lists[0]))
    
    print(len(id_lists))
    print(type(id_lists[0]))
    
    df = pd.DataFrame(data=label_lists)
    df['id'] = id_lists
    df['cdr'] = cdr_lists

    df.to_csv('/home/d1353/no_backup/d1353/Code/ResNet34/Shuffled_labels_evalhealthy.csv')
    
    '''
        Eval AD patients
    '''
    label_lists = []
    id_lists = []
    cdr_lists = []
    
    y1 = patientInfo.loc[patientInfo['cdr'] != 0]
    y3 = list(y1['age'].unique())
    y3.sort()
    
    c = []
    for i in y3:
        c.append(y1.loc[y1['age'] == i,'id'].values.tolist())
        
    #print(c)
    print(len(c),len(y3))

    final_adpatients = []
    for j in range(0,len(c)):
        if len(c[j]) < 2:
            final_adpatients.append(c[j])
        else:
            final_adpatients.append(random.sample(c[j], 2))

    final_adpatients = [item for sublist in final_adpatients for item in sublist]
    print(len(final_adpatients))
    
    shuffle(final_adpatients)
    
    eval_ad = [path for path in pat_path_list if path[path_length:-9] in final_adpatients]
    
    for id in final_adpatients:    
        label_lists.append(y1.loc[y1['id'] == id,'age'].values[0])
        id_lists.append(y1.loc[y1['id'] == id,'id'].values[0])
        cdr_lists.append(y1.loc[y1['id'] == id,'cdr'].values[0])
        
    print(len(label_lists))
    print(type(label_lists[0]))
    
    print(len(id_lists))
    print(type(id_lists[0]))
    
    df = pd.DataFrame(data=label_lists)
    df['id'] = id_lists
    df['cdr'] = cdr_lists

    df.to_csv('/home/d1353/no_backup/d1353/Code/ResNet34/Shuffled_labels_evalAD.csv')
    
    return eval_healthy, eval_ad