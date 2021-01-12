import os
import time
import sys
import pickle
import matplotlib.pyplot as plt
import math
import pandas as pd
import tensorflow as tf
import numpy as np
import tensorflow.keras
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.backend as K

from datetime import datetime
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# own scripts
import get_files_oasis as get_train_eval_files
import Res34_model as base
import gen_pairs_oasis as generator

def train():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    # Divide the entire dataset into training and validation set
    train_path = "/home/d1353/no_backup/d1353/OASIS_DS/"
    test_path = "/home/d1353/no_backup/d1353/OASIS_test/"
    batch_size = 16
    num_parallel_calls = 4
    n_epoch = 50
    latent_dim = 128

    train_healthy, train_ad = get_train_eval_files.prepare_train_files(train_path)
    eva_healthy, eva_ad = get_train_eval_files.prepare_eval_files(test_path)
    
    train_path_length = len(train_path)
    print(train_path_length)
    
    test_path_length = len(test_path)
    print(test_path_length)
    
    total_patient_healthy1 = []
    # Get the validation patients names
    for i in range(len(eva_healthy)):
        validation_patient_number1 = eva_healthy[i][test_path_length:-9]
        total_patient_healthy1.append(validation_patient_number1)

    total_patient_healthy2 = []
    # Get the training patients names
    for i in range(len(train_healthy)):
        validation_patient_number2 = train_healthy[i][train_path_length:-9]
        total_patient_healthy2.append(validation_patient_number2)
        
    total_patient_AD1 = []
    # Get the validation patients names
    for i in range(len(eva_ad)):
        validation_patient_number1 = eva_ad[i][test_path_length:-9]
        total_patient_AD1.append(validation_patient_number1)

    total_patient_AD2 = []
    # Get the training patients names
    for i in range(len(train_ad)):
        validation_patient_number2 = train_ad[i][train_path_length:-9]
        total_patient_AD2.append(validation_patient_number2)

    print('Length of training patients is: ', len(train_healthy)+len(train_ad))
    print('The training healthy patients are: ', total_patient_healthy2)
    print('Length of validation patients is: ', len(eva_healthy)+len(eva_ad))
    print('The validation healthy patients are: ', total_patient_healthy1)
    print('The training ad patients are: ', total_patient_AD2)
    print('The validation ad patients are: ', total_patient_AD1)

    print('Batch size : ', batch_size)

    print('-' * 75)
    print(' Model\n')
    
    def accuracy(y_true, y_pred):
        return K.mean(y_pred[:,0] < y_pred[:,1])
    
    def triplet_loss(y_true, y_pred):

        alpha = 0.2
        basic_loss = K.square(y_pred[:,0]) - K.square(y_pred[:,1]) + alpha
        return K.mean(K.maximum(basic_loss, K.constant(0)))
        
    encoder = base.model_create()

    positive_input = tf.keras.Input(shape=(121, 145, 1), name="positive_input")
    anchor_input = tf.keras.Input(shape=(121, 145, 1), name="anchor_input")
    negative_input = tf.keras.Input(shape=(121, 145, 1), name="negative_input")

    encoded_p = encoder(positive_input)
    encoded_a = encoder(anchor_input)
    encoded_n = encoder(negative_input)
    
    #for triplet loss
    pos_distance = K.sqrt(K.maximum(K.sum(K.square(encoded_p - encoded_a), axis=1, keepdims=True), K.epsilon())) #K.sum(K.square(encoded_p - encoded_a), axis=1)
    neg_distance = K.sqrt(K.maximum(K.sum(K.square(encoded_a - encoded_n), axis=1, keepdims=True), K.epsilon())) #K.sum(K.square(encoded_a - encoded_n),axis=1)
    
    # This lambda layer simply stacks outputs so both distances are available to the objective
    stacked_dists = tf.keras.layers.Lambda(lambda vects: K.stack(vects, axis=1), name='stacked_dists')([pos_distance, neg_distance])

    siamese = Model(inputs=[anchor_input, positive_input, negative_input], outputs=stacked_dists,
                    name='triplet')
    siamese.compile(loss=triplet_loss, optimizer=optimizers.Adam(lr=0.00001), metrics = [accuracy])
    
    print(siamese.summary())

    logdir = "/home/d1353/no_backup/d1353/Code/ResNet34/logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    
    model_file = "/home/d1353/no_backup/d1353/Code/ResNet34/AD.tf"

    print('-' * 75)
    print(' Training...')
    
    def get_callbacks():
        callbacks = list()
        #Save the log file
        callbacks.append(tensorboard_callback)
        #Save the model
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(model_file, monitor='val_accuracy', save_best_only=True, mode='max'))
        #Stop training in case of validation error increase
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.005, patience=5, verbose=0, mode='auto', baseline=None, restore_best_weights=False))
        #Reduce LR On Plateau
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, cooldown=2))

        return callbacks
    
    # For the training patients    
    tr_healthy, tr_healthy_ids = generator.tfdata_generator_2Dslices(file_lists=train_healthy, 
                                                                    patient_id=total_patient_healthy2,
                                                                    num_parallel_calls=num_parallel_calls,
                                                                    case='train')
    chunked_data_ht = generator.div_based_on_chunk(tr_healthy, len(train_healthy))
                                                                    
    tr_ad, tr_ad_ids = generator.tfdata_generator_2Dslices(file_lists=train_ad, 
                                                            patient_id=total_patient_AD2,
                                                            num_parallel_calls=num_parallel_calls,
                                                            case='train')                                                                
    chunked_data_adt = generator.div_based_on_chunk(tr_ad, len(train_ad))
    
    tr_triplets, dummy_tr_label = generator.gen_pairs(tr_healthy_ids, tr_ad_ids)
    
    # For the validation patients    
    val_healthy, val_healthy_ids = generator.tfdata_generator_2Dslices(file_lists=eva_healthy, 
                                                                    patient_id=total_patient_healthy1,
                                                                    num_parallel_calls=num_parallel_calls,
                                                                    case='train')
    chunked_data_hv = generator.div_based_on_chunk(val_healthy, len(val_healthy))
                                                                    
    val_ad, val_ad_ids = generator.tfdata_generator_2Dslices(file_lists=train_ad, 
                                                            patient_id=total_patient_AD1,
                                                            num_parallel_calls=num_parallel_calls,
                                                            case='train')                                                                
    chunked_data_adv = generator.div_based_on_chunk(val_ad, len(val_ad))
    
    val_triplets, dummy_val_label = generator.gen_pairs(val_healthy_ids, val_ad_ids)
    
    steps_per_epoch = math.ceil(len(dummy_tr_label) / batch_size) + 1
    validata_steps = math.ceil(len(dummy_val_label) / batch_size) + 1 
    
    print('Expected training steps: ', steps_per_epoch)
    print('Expected validation_steps: ', validata_steps)
    
    tr_generator = generator.gen_batch_run(chunked_data_ht, chunked_data_adt, tr_triplets, dummy_tr_label, batch_size)
    val_generator = generator.gen_batch_run(chunked_data_hv, chunked_data_adv,val_triplets, dummy_val_label, batch_size)
    
    result = siamese.fit_generator(tr_generator, steps_per_epoch=steps_per_epoch, epochs=n_epoch, validation_data=val_generator, validation_steps=validata_steps, callbacks= get_callbacks())
    print('\nhistory dict:', result.history)   
    
    #print('Siamese model save')
    #siamese.save('/home/d1353/no_backup/d1353/Code/ResNet34/siamese')

    #print('encoder model save')
    #encoder.save('/home/d1353/no_backup/d1353/Code/ResNet34/encoder')


if __name__ == '__main__':
    train()
