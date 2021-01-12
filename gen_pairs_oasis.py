import tensorflow as tf
import convert_tf
import numpy as np
import pickle
import pandas as pd
import random 
from random import shuffle

age_list = []
id_list = [] 


# Accepts a list of files of patients data and returns a tfrecord dataset
def tfdata_generator_2Dslices(file_lists, patient_id, num_parallel_calls, case):

    filenames = tf.data.TFRecordDataset(file_lists)

    ids = [[c] * 5 for c in patient_id]
    ids = [item for sublist in ids for item in sublist]

    if case == 'encode':
        print('append called')
        id_list.append(ids)
        print(len(id_list))

    dataset = filenames.map(map_func=lambda a: (convert_tf.parse_function_image(a)),
                            num_parallel_calls=num_parallel_calls)

    slice_collection = []

    axial_slices = [82, 85, 88, 91, 94]

    for element in dataset:
        # print('chunking data number'+str(i))
        for j in axial_slices:
            x = element[:, :, j]
            # print(x.shape)
            slice_collection.append(x.numpy().reshape(121, 145, 1))

    print(len(slice_collection))

    return slice_collection, ids


def div_based_on_chunk(data, length):

    chunk_data = np.zeros((5, length, 121, 145, 1))
    
    for i, datum in enumerate(data):
        j = i//5
        k = i%5
        chunk_data[k][j] = datum
        
    return(chunk_data)


def gen_pairs(ids, ad_ids):
    
    triplets = []
    done_id = []
       
    for i in range(0, len(ids), 5):
        print('patient id', ids[i])
        print('i', i)
         
        p_id = i//5
        print('p_id', p_id)
        done_id.append(p_id)
       
        for j in range(0,5):
        
            '''
                same pairs
            '''
            same_indices = []
            same_chunks = []
            for k in range(0, len(ids)):
                if k%5 == j and k//5 not in done_id:
                    same_indices.append(k)

            if len(same_indices) < 15:
                same_chunks.append(same_indices)
            else:                
                same_chunks.append(random.sample(same_indices, 15))
            
            same_chunks = [item for sublist in same_chunks for item in sublist]
            print('same chunks for', i, len(same_chunks))
            print(same_chunks)
            
            '''
                diff pairs
            '''
            diff_indices = []
            diff_chunks = []     
            for k in range(0, len(ad_ids)):
                if k%5 == j:
                    diff_indices.append(k)
                    
            if len(diff_indices) < 15:
                diff_chunks.append(diff_indices)
            else:                    
                diff_chunks.append(random.sample(diff_indices, 15))
            
            diff_chunks = [item for sublist in diff_chunks for item in sublist]  
            print('diff chunks for', i, len(diff_chunks))
            print(diff_chunks)            

            for x,y in zip(same_chunks,diff_chunks): 
                p_id1 = x//5
                p_id2 = y//5
                temp0 = str(j)+str(':')+str(p_id)+str('$h') 
                temp1 = str(j)+str(':')+str(p_id1)+str('$h')
                temp2 = str(j)+str(':')+str(p_id2)+str('$h') 
                triplets +=[[temp0, temp1, temp2]] 
    
    n_len_trip = len(triplets) 
    print('len of normal anchor triplets', len(triplets))
    
    done_id = []
    for i in range(0,len(ad_ids),5):
        print('patient id', ad_ids[i])
        print('i', i)
        
        p_id = i//5
        print('p_id', p_id)
        done_id.append(p_id)
       
        for j in range(0,5):
        
            '''
                same pairs
            '''
            same_indices = []
            same_chunks = []
            for k in range(0, len(ad_ids)):
                if k%5 == j and k//5 not in done_id:
                    same_indices.append(k)
            
            if len(same_indices) < 15:
                same_chunks.append(same_indices)
            else:            
                same_chunks.append(random.sample(same_indices, 15))
            
            same_chunks = [item for sublist in same_chunks for item in sublist]
            print('same chunks for', i, len(same_chunks))
            print(same_chunks)
            
            '''
                diff pairs
            '''
            diff_indices = []
            diff_chunks = []     
            for k in range(0, len(ids)):
                if k%5 == j:
                    diff_indices.append(k)

            if len(diff_indices) < 15:
                diff_chunks.append(diff_indices)
            else:                         
                diff_chunks.append(random.sample(diff_indices, 15))
            
            diff_chunks = [item for sublist in diff_chunks for item in sublist]  
            print('diff chunks for', i, len(diff_chunks))
            print(diff_chunks)            

            for x,y in zip(same_chunks,diff_chunks): 
                p_id1 = x//5
                p_id2 = y//5
                temp0 = str(j)+str(':')+str(p_id)+str('$a')  
                temp1 = str(j)+str(':')+str(p_id1)+str('$a')  
                temp2 = str(j)+str(':')+str(p_id2)+str('$a') 
                triplets +=[[temp0, temp1, temp2]] 
                
    print('len of all triplets', len(triplets))
    a_len_trip = len(triplets) - n_len_trip

    dummy_label = [0]*len(triplets)
    
    return triplets, dummy_label

    
def gen_batch_run(chunked_data_h, chunked_data_ad, triplets, targets, batch_size):
    
    n_samples = len(targets)
    print('n_samples', n_samples)
    
    while True:  
        shuffle(triplets)
        
        for offset in range(0, n_samples, batch_size):
            batch_triplets = triplets[offset:offset+batch_size]
            x_train = []
            y_train = []
            
            for i, batch_t in enumerate(batch_triplets):
                #print('triplet',batch_t)
                chunk_num = int(batch_t[0].split(':')[0])
                #print('chunk_num',chunk_num)
                pid = int((batch_t[0].split(':')[1]).split('$')[0])
                #print('pid1',pid)
                pid1 = int((batch_t[1].split(':')[1]).split('$')[0])
                #print('pid2',pid1)
                pid2 = int((batch_t[2].split(':')[1]).split('$')[0])
                #print('pid2',pid2)
                m = batch_t[0].split('$')[1]
                
                assert int(batch_t[0].split(':')[0]) == int(batch_t[1].split(':')[0]) 
                assert int(batch_t[0].split(':')[0]) == int(batch_t[2].split(':')[0])
                
                assert batch_t[0].split('$')[1] == batch_t[1].split('$')[1] 
                assert batch_t[0].split('$')[1] == batch_t[2].split('$')[1] 
                
                #retrieve data from chunked data list
                if m == 'h':
                    temp0 = chunked_data_h[chunk_num][pid].reshape(121, 145, 1)
                    temp1 = chunked_data_h[chunk_num][pid1].reshape(121, 145, 1)
                    temp2 = chunked_data_ad[chunk_num][pid2].reshape(121, 145, 1)
                elif m == 'a':
                    temp0 = chunked_data_ad[chunk_num][pid].reshape(121, 145, 1)
                    temp1 = chunked_data_ad[chunk_num][pid1].reshape(121, 145, 1)
                    temp2 = chunked_data_h[chunk_num][pid2].reshape(121, 145, 1)
                    
                x_train += [[temp0, temp1, temp2]]
                y_train.append(targets[offset + i])
            
            x_train = np.asarray(x_train)
            y_train = np.asarray(y_train)
            
            yield [x_train[:,0] ,x_train[:,1], x_train[:,2]], y_train