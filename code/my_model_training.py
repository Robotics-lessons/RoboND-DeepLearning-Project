import os
import glob
import sys
import tensorflow as tf
import time
from scipy import misc
import numpy as np
from datetime import datetime, date
from tensorflow.contrib.keras.python import keras
from tensorflow.contrib.keras.python.keras import layers, models
import logging
from tensorflow import image

from utils import scoring_utils
from utils.separable_conv2d import SeparableConv2DKeras, BilinearUpSampling2D
from utils import data_iterator
from utils import plotting_tools 
from utils import model_tools


def separable_conv2d_batchnorm(input_layer, filters, strides=1):
    output_layer = SeparableConv2DKeras(filters=filters,kernel_size=3, strides=strides,
                             padding='same', activation='relu')(input_layer)
    
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer

def conv2d_batchnorm(input_layer, filters, kernel_size=3, strides=1):
    output_layer = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, 
                      padding='same', activation='relu')(input_layer)
    
    output_layer = layers.BatchNormalization()(output_layer) 
    return output_layer

def bilinear_upsample(input_layer):
    try:
        output_layer = BilinearUpSampling2D((2,2))(input_layer)
    except:
        logging.error("Unexpected error:", sys.exc_info()[0])
    return output_layer

def encoder_block(input_layer, filters, strides):
    
    # TODO Create a separable convolution layer using the separable_conv2d_batchnorm() function.
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides=strides)
#    print('output_layer Shape in encoder_block: {}'.format(output_layer.shape))  
    
    return output_layer

def decoder_block(small_ip_layer, large_ip_layer, filters):
    
    # TODO Upsample the small input layer using the bilinear_upsample() function.
    upsampled_layer = bilinear_upsample(small_ip_layer)

#    print('upsampled_layer in decoder_block:', upsampled_layer)
#    print('large_ip_layer in decoder_block:', large_ip_layer)    
    # TODO Concatenate the upsampled and large input layers using layers.concatenate
    if large_ip_layer != None:
        concatenate_layer = layers.concatenate([upsampled_layer, large_ip_layer])
    else:
        concatenate_layer = upsampled_layer
#    print('concatenate_layer in decoder_block:', concatenate_layer)    
    # TODO Add some number of separable convolution layers
    output_layer = separable_conv2d_batchnorm(concatenate_layer, filters)
#    output_layer = separable_conv2d_batchnorm(output_layer, 3)
#    print('output_layer last Shape in decoder_block: {}'.format(output_layer.shape))    
    return output_layer

def fcn_model(inputs, num_classes, filters=32):
    logging.info('inputs : {}'.format(inputs.shape))       
    # TODO Add Encoder Blocks. 
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    input_layer1 = encoder_block(inputs, filters, 2)
    input_layer2 = encoder_block(input_layer1, filters*2, 2)  
    input_layer3 = encoder_block(input_layer2, filters*4, 2) 
    input_layer4 = encoder_block(input_layer3, filters*8, 2) 
    input_layer5 = encoder_block(input_layer4, filters*16, 2) 
    input_layer6 = encoder_block(input_layer5, filters*32, 2) 
    # print out
    logging.info('input_layer 1 : {}'.format(input_layer1.shape))
    logging.info('input_layer 2 : {}'.format(input_layer2.shape))
    logging.info('input_layer 3 : {}'.format(input_layer3.shape))
    logging.info('input_layer 4 : {}'.format(input_layer4.shape))
    logging.info('input_layer 5 : {}'.format(input_layer5.shape))
    logging.info('input_layer 6 : {}'.format(input_layer6.shape))
    
    # TODO Add 1x1 Convolution layer using conv2d_batchnorm().
#    small_ip_layer = conv2d_batchnorm(input_layer5, filters*32, kernel_size=1, strides=1)
    small_ip_layer = conv2d_batchnorm(input_layer6, filters*64, kernel_size=1, strides=1)
    logging.info('small_ip_layer : {}'.format(small_ip_layer.shape)) 
    
    # TODO: Add the same number of Decoder Blocks as the number of Encoder Blocks
    output_layer6 = decoder_block(small_ip_layer, input_layer5, filters*32) 
    logging.info('output_layer 6: {}'.format(output_layer6.shape)) 
    output_layer5 = decoder_block(output_layer6, input_layer4, filters*16) 
#    output_layer5 = decoder_block(small_ip_layer, input_layer4, filters*16) 
    logging.info('output_layer 5: {}'.format(output_layer5.shape)) 
    output_layer4 = decoder_block(output_layer5, input_layer3, filters*8)  
    logging.info('output_layer 4: {}'.format(output_layer4.shape)) 
    output_layer3 = decoder_block(output_layer4, input_layer2, filters*4)  
    logging.info('output_layer 3: {}'.format(output_layer3.shape)) 
    output_layer2 = decoder_block(output_layer3, input_layer1, filters*2) 
    logging.info('output_layer 2: {}'.format(output_layer2.shape))
    output_layer1 = decoder_block(output_layer2, None, 3) 
    logging.info('output_layer 1: {}'.format(output_layer1.shape)) 
    
    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(output_layer1)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    filename='../logs/training_model_' + datetime.now().strftime('model_%Y-%m-%d-%H-%M-%S') + '.log',
                    filemode='w')
    image_hw =  256
    image_shape = (image_hw, image_hw, 3)
    inputs = layers.Input(image_shape)
    num_classes = 3
    logging.info('================================================================')
# Call fcn_model()
    output_layer = fcn_model(inputs, num_classes)

    arr = []
    learning_rates = [0.001, 0.0005, 0.0002]
    batch_size = 32
    num_epochs = 200
    steps_per_epoch =  len([name for name in os.listdir('../data/train/images')]) // (batch_size*2) # 200
    validation_steps = len([name for name in os.listdir('../data/validation/images')]) // batch_size #50
    workers = 1

    min_loss_value = 0.01
    logging.info('min_loss_value = %f' % min_loss_value)
    step_stop_loop_after_best_score = 3
    logging.info('step_stop_loop_after_best_score = %d' % step_stop_loop_after_best_score)
    step_stop_loop_after_lowest_loss = 5
    logging.info('step_stop_loop_after_lowest_loss = %d' % step_stop_loop_after_lowest_loss)
    best_score_num_epoch = 0
    lowest_loss_num_epoch = 0


    for learning_rate in learning_rates:
        lowest_loss = 10.0
        score = 0.35
        logging.info('================================================================')
        logging.info('learning rate = %f' % learning_rate)
        logging.info('batch_size = %d' % batch_size)
        logging.info('num_epochs = %d' % num_epochs)
        logging.info('steps_per_epoch = %d' % steps_per_epoch)
        logging.info('validation_steps = %d' % validation_steps)
        logging.info('workers = %d' % workers)

# Define the Keras model and compile it for training
        model = models.Model(inputs=inputs, outputs=output_layer)

        model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='categorical_crossentropy')

      # Train pics
        train_iter = data_iterator.BatchIteratorSimple(batch_size=batch_size,
                                                       data_folder=os.path.join('..', 'data', 'train'),
                                                       image_shape=image_shape,
                                                       shift_aug=True)
            
        # Validation pics
        val_iter = data_iterator.BatchIteratorSimple(batch_size=batch_size,
                                                     data_folder=os.path.join('..', 'data', 'validation'),
                                                     image_shape=image_shape)
 
        logging.info('score = %f' % score)
        t1 = time.time()
        for i in range(num_epochs):
        
  
        # Logs
        #    logger_cb = plotting_tools.LoggerPlotter()
        #    callbacks = [logger_cb]
        
        # Fit into the model
            history = model.fit_generator(train_iter,
                            steps_per_epoch = steps_per_epoch, # the number of batches per epoch,
                            epochs = 1, # the number of epochs to train for,
                            validation_data = val_iter, # validation iterator
                            validation_steps = validation_steps, # the number of batches to validate on
                            #callbacks=callbacks,
                            workers = workers)
            loss_value = history.history['loss'][0]
            validation_loss = history.history['val_loss'][0]
            logging.info('epoch number = %d, loss = %f, val_loss = %f' % (i + 1, loss_value, validation_loss))
            if loss_value < lowest_loss:
                lowest_loss = loss_value
                lowest_loss_num_epoch = i
            elif i - lowest_loss_num_epoch >= step_stop_loop_after_lowest_loss:
                logging.info('epoch = %d, lowest_loss_num_epoch = %d' % (i, lowest_loss_num_epoch))
                logging.info('stop training, because loss value does not decrease in %d epochs' % step_stop_loop_after_lowest_loss)
                break
            if min_loss_value > loss_value:
        #### Evaluate ####
        
        # Run provided tests
        
                run_num = 'run_1'        
        
                val_with_targ, pred_with_targ = model_tools.write_predictions_grade_set(model,
                                                run_num,'patrol_with_targ', 'sample_evaluation_data') 
        
                val_no_targ, pred_no_targ = model_tools.write_predictions_grade_set(model, 
                                                run_num,'patrol_non_targ', 'sample_evaluation_data') 
        
                val_following, pred_following = model_tools.write_predictions_grade_set(model,
                                                run_num,'following_images', 'sample_evaluation_data')
    
#        print("Images while following the target")
#        im_files = plotting_tools.get_im_file_sample('sample_evaluation_data','following_images', run_num) 
#        for i in range(3):
#            im_tuple = plotting_tools.load_images(im_files[i])
#            plotting_tools.show_images(im_tuple)
            
            
#        print("Images while at patrol without target")
#        im_files = plotting_tools.get_im_file_sample('sample_evaluation_data','patrol_non_targ', run_num) 
#        for i in range(3):
#            im_tuple = plotting_tools.load_images(im_files[i])
#            plotting_tools.show_images(im_tuple)
         
            
#        print("Images while at patrol with target")
#        im_files = plotting_tools.get_im_file_sample('sample_evaluation_data','patrol_with_targ', run_num) 
#        for i in range(3):
#            im_tuple = plotting_tools.load_images(im_files[i])
#            plotting_tools.show_images(im_tuple)

                logging.info("Quad behind the target")
                true_pos1, false_pos1, false_neg1, iou1 = scoring_utils.score_run_iou(val_following, pred_following)
        
                logging.info("Target not visible")
                true_pos2, false_pos2, false_neg2, iou2 = scoring_utils.score_run_iou(val_no_targ, pred_no_targ)
        
                logging.info("Target far away")
                true_pos3, false_pos3, false_neg3, iou3 = scoring_utils.score_run_iou(val_with_targ, pred_with_targ)
        
        # Sum all the true positives, etc from the three datasets to get a weight for the score
                true_pos = true_pos1 + true_pos2 + true_pos3
                false_pos = false_pos1 + false_pos2 + false_pos3
                false_neg = false_neg1 + false_neg2 + false_neg3
        
                weight = true_pos/(true_pos+false_neg+false_pos)
                logging.info('weight = %f ' % weight)
        
        
        # The IoU for the dataset that never includes the hero is excluded from grading
                final_IoU = (iou1 + iou3)/2
                logging.info("IoU no hero - ", final_IoU)
              
        # And the final grade score is 
                final_score = final_IoU * weight
                logging.info("Final Grade - ", final_score)

                if final_score > score:
            # Save model with best score
                    weight_file_name = str(learning_rate) + '_' + str(final_score) + '_' + datetime.now().strftime('model_%Y-%m-%d-%H-%M-%S') + '.h5'
                    logging.info('Model weight file name = %s' % weight_file_name)
                    model_tools.save_network(your_model=model, your_weight_filename=weight_file_name)
                    best_score_num_epoch = i + 1          
                    score = final_score
                    arr = [best_score_num_epoch, score,final_IoU,weight,weight_file_name]
                elif i - best_score_num_epoch >= step_stop_loop_after_best_score:
                    logging.info('Score no long improving at epoch = %d' % i)
                    break
            
        logging.info('Best score = ', score)
        logging.info('Data ---  {}'.format(arr))
        t2 = time.time()
        logging.info("Time: %0.2fs" % (t2 - t1))
    


