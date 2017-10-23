[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)

# Follow Me Project #

In this project, you will train a deep neural network to identify and track a target in simulation. So-called “follow me” applications like this are key to many fields of robotics and the very same techniques you apply here could be extended to scenarios like advanced cruise control in autonomous vehicles or human-robot collaboration in industry.

[image_0]: ./docs/misc/sim_screenshot.png 
![alt text][image_0] 

## Setup Instructions ##
**Clone the repository**

    $ git clone https://github.com/udacity/RoboND-DeepLearning.git


**Download the data**

Save the following three files into the data folder of the cloned repository. 

[Training Data](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Lab/train.zip) 

[Validation Data](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Lab/validation.zip)

[Sample Evaluation Data](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Project/sample_evaluation_data.zip)

**Download the QuadSim binary**

To interface your neural net with the QuadSim simulator, you must use a version QuadSim that has been custom tailored for this project. The previous version that you might have used for the Controls lab will not work.

The simulator binary can be downloaded [here](https://github.com/udacity/RoboND-DeepLearning/releases/latest)

**Install Dependencies**

You'll need Python 3 and Jupyter Notebooks installed to do this project.  The best way to get setup with these if you are not already is to use Anaconda following along with the [RoboND-Python-Starterkit](https://github.com/udacity/RoboND-Python-StarterKit).

If for some reason you choose not to use Anaconda, you must install the following frameworks and packages on your system:
* Python 3.x
* Tensorflow 1.2.1
* NumPy 1.11
* SciPy 0.17.0
* eventlet 
* Flask
* h5py
* PIL
* python-socketio
* scikit-image
* transforms3d
* PyQt4/Pyqt5

## Implement the Segmentation Network
 1. Download the training dataset from above and extract to the project `data` directory.
 2. Implement your solution in model_training.ipynb
 3. Train the network locally, or on [AWS](https://classroom.udacity.com/nanodegrees/nd209/parts/09664d24-bdec-4e64-897a-d0f55e177f09/modules/cac27683-d5f4-40b4-82ce-d708de8f5373/lessons/197a058e-44f6-47df-8229-0ce633e0a2d0/concepts/27c73209-5d7b-4284-8315-c0e07a7cd87f?contentVersion=1.0.0&contentLocale=en-us).
 4. Continue to experiment with the training data and network until you attain the score you desire.
 5. Once you are comfortable with performance on the training dataset, see how it performs in live simulation!

## Collecting Training Data ##
A simple training dataset has been provided in this project's repository. This dataset will allow you to verify that your segmentation network is semi-functional. However, if your interested in improving your score,you may want to collect additional training data. To do it, please see the following steps.

The data directory is organized as follows:
```
data/runs - contains the results of prediction runs
data/train/images - contains images for the training set
data/train/masks - contains masked (labeled) images for the training set
data/validation/images - contains images for the validation set
data/validation/masks - contains masked (labeled) images for the validation set
data/weights - contains trained TensorFlow models

data/raw_sim_data/train/run1
data/raw_sim_data/validation/run1
```

### Training Set ###
 1. Run QuadSim
 2. Click the `DL Training` button
 3. Set patrol points, path points, and spawn points. 
 4. With the simulator running, press "r" to begin recording.
 5. In the file selection menu navigate to the `data/raw_sim_data/train/run1` directory
 6. **optional** to speed up data collection, press "9" (1-9 will slow down collection speed)
 7. When you have finished collecting data, hit "r" to stop recording.
 8. To reset the simulator, hit "`<esc>`"
 9. To collect multiple runs create directories `data/raw_sim_data/train/run2`, `data/raw_sim_data/train/run3` and repeat the above steps.


### Validation Set ###
To collect the validation set, repeat both sets of steps above, except using the directory `data/raw_sim_data/validation` instead rather than `data/raw_sim_data/train`.

### Image Preprocessing ###
Before the network is trained, the images first need to be undergo a preprocessing step. The preprocessing step transforms the depth masks from the sim, into binary masks suitable for training a neural network. It also converts the images from .png to .jpeg to create a reduced sized dataset, suitable for uploading to AWS. 
To run preprocessing:
```
$ python preprocess_ims.py
```
## Training, Predicting and Scoring ##
With your training and validation data having been generated or downloaded from the above section of this repository, you are free to begin working with the neural net.

**Note**: Training CNNs is a very compute-intensive process. If your system does not have a recent Nvidia graphics card, with [cuDNN](https://developer.nvidia.com/cudnn) and [CUDA](https://developer.nvidia.com/cuda) installed , you may need to perform the training step in the cloud. Instructions for using AWS to train your network in the cloud may be found [here](https://classroom.udacity.com/nanodegrees/nd209/parts/09664d24-bdec-4e64-897a-d0f55e177f09/modules/cac27683-d5f4-40b4-82ce-d708de8f5373/lessons/197a058e-44f6-47df-8229-0ce633e0a2d0/concepts/27c73209-5d7b-4284-8315-c0e07a7cd87f?contentVersion=1.0.0&contentLocale=en-us)

### Training your Model ###
**Prerequisites**
- Training data is in `data` directory
- Validation data is in the `data` directory
- The folders `data/train/images/`, `data/train/masks/`, `data/validation/images/`, and `data/validation/masks/` should exist and contain the appropriate data

To train complete the network definition in the `model_training.ipynb` notebook and then run the training cell with appropriate hyperparameters selected.

After the training run has completed, your model will be stored in the `data/weights` directory as an [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) file, and a configuration_weights file. As long as they are both in the same location, things should work. 

**Important Note** the *validation* directory is used to store data that will be used during training to produce the plots of the loss, and help determine when the network is overfitting your data. 

The **sample_evalution_data** directory contains data specifically designed to test the networks performance on the FollowME task. In sample_evaluation data are three directories each generated using a different sampling method. The structure of these directories is exactly the same as `validation`, and `train` datasets provided to you. For instance `patrol_with_targ` contains an `images` and `masks` subdirectory. If you would like to the evaluation code on your `validation` data a copy of the it should be moved into `sample_evaluation_data`, and then the appropriate arguments changed to the function calls in the `model_training.ipynb` notebook.

The notebook has examples of how to evaulate your model once you finish training. Think about the sourcing methods, and how the information provided in the evaluation sections relates to the final score. Then try things out that seem like they may work. 

## Scoring ##

To score the network on the Follow Me task, two types of error are measured. First the intersection over the union for the pixelwise classifications is computed for the target channel. 

In addition to this we determine whether the network detected the target person or not. If more then 3 pixels have probability greater then 0.5 of being the target person then this counts as the network guessing the target is in the image. 

We determine whether the target is actually in the image by whether there are more then 3 pixels containing the target in the label mask. 

Using the above the number of detection true_positives, false positives, false negatives are counted. 

**How the Final score is Calculated**

The final score is the pixelwise `average_IoU*(n_true_positive/(n_true_positive+n_false_positive+n_false_negative))` on data similar to that provided in sample_evaulation_data

**Ideas for Improving your Score**

Collect more data from the sim. Look at the predictions think about what the network is getting wrong, then collect data to counteract this. Or improve your network architecture and hyperparameters. 

**Obtaining a Leaderboard Score**

Share your scores in slack, and keep a tally in a pinned message. Scores should be computed on the sample_evaluation_data. This is for fun, your grade will be determined on unreleased data. If you use the sample_evaluation_data to train the network, it will result in inflated scores, and you will not be able to determine how your network will actually perform when evaluated to determine your grade.

## Experimentation: Testing in Simulation
1. Copy your saved model to the weights directory `data/weights`.
2. Launch the simulator, select "Spawn People", and then click the "Follow Me" button.
3. Run the realtime follower script

    $ python follower.py my_amazing_model.h5


## Model Architecture ##
Using fully convolutional network (FCN) to build this model architecture. 

1. 6 encoder_block for encoder layers 
2. the 1x1 convolution layer
3. 6 decoder_block for decoder layers.
4. 4 skip connections between the encoder and decoder layers.

[image_1]: ./docs/misc/FCNArchitecture.png 
![alt text][image_1]

### Separable Convolutions

The Encoder for FCN will essentially require separable convolution layers. The 1x1 convolution layer in the FCN, however, is a regular convolution. Implementations for both are provided below for using. Each includes batch normalization with the ReLU activation function applied to the layers.


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


### Bilinear Upsampling

The following helper function implements the bilinear upsampling layer. Upsampling by a factor of 2 is generally recommended. Upsampling is used in the decoder block of the FCN.
 
    def bilinear_upsample(input_layer):
        output_layer = BilinearUpSampling2D((2,2))(input_layer)
        return output_layer


## Build the Model

   Build the FCN consisting of encoder block(s), a 1x1 convolution, and decoder block(s). This step requires experimentation with different numbers of layers and filter sizes to build your model.
   
### Encoder Block

Create an encoder block that includes a separable convolution layer using the separable_conv2d_batchnorm() function. The filters parameter defines the size or depth of the output layer. For example, 32 or 64.


    def encoder_block(input_layer, filters, strides):
    
    # TODO Create a separable convolution layer using the separable_conv2d_batchnorm() function.
        output_layer = separable_conv2d_batchnorm(input_layer, filters, strides=strides)    
        return output_layer


### Decoder Block

The decoder block is comprised of three parts:

   A bilinear upsampling layer using the upsample_bilinear() function. The current recommended factor for upsampling is set to 2.
   A layer concatenation step. This step is similar to skip connections. The data will be concatenated by the upsampled small_ip_layer and the large_ip_layer.
   Some (one or two) additional separable convolution layers to extract some more spatial information from prior layers.
   If condition statement will check input large_ip_layer value is None or not, if it is None, then skip the concatenating function.

    def decoder_block(small_ip_layer, large_ip_layer, filters):
    
        # TODO Upsample the small input layer using the bilinear_upsample() function.
        upsampled_layer = bilinear_upsample(small_ip_layer)

        # TODO Concatenate the upsampled and large input layers using layers.concatenate
        if large_ip_layer != None:
            concatenate_layer = layers.concatenate([upsampled_layer, large_ip_layer])
        else:
            concatenate_layer = upsampled_layer
   
        # TODO Add some number of separable convolution layers
        output_layer = separable_conv2d_batchnorm(concatenate_layer, filters)
 
        return output_layer


### Model Function

There are three steps to build a FCN architecture:

   1. Add encoder blocks to build the encoder layers. 
   2. Add a 1x1 Convolution layer using the conv2d_batchnorm() function. Remember that 1x1 Convolutions require a kernel and stride of 1.
   3. Add decoder blocks for the decoder layers.
   
    def fcn_model(inputs, num_classes, filters=32):
        print('inputs : ',inputs)     
        # TODO Add Encoder Blocks. 
        # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
        input_layer1 = encoder_block(inputs, filters, 2)
        input_layer2 = encoder_block(input_layer1, filters*2, 2)  
        input_layer3 = encoder_block(input_layer2, filters*4, 2) 
        input_layer4 = encoder_block(input_layer3, filters*8, 2) 
        input_layer5 = encoder_block(input_layer4, filters*16, 2) 
        input_layer6 = encoder_block(input_layer5, filters*32, 2) 

        print('input_layer 1 :',input_layer1)
        print('input_layer 2 :',input_layer2)
        print('input_layer 3 :',input_layer3)
        print('input_layer 4 :',input_layer4) 
        print('input_layer 5 :',input_layer5)
        print('input_layer 6 :',input_layer6)
    
        # TODO Add 1x1 Convolution layer using conv2d_batchnorm().
        small_ip_layer = conv2d_batchnorm(input_layer6, filters*64, kernel_size=1, strides=1)
        print('small_ip_layer :',small_ip_layer) 
    
        # TODO: Add the same number of Decoder Blocks as the number of Encoder Blocks
        output_layer6 = decoder_block(small_ip_layer, input_layer5, filters*32) 
        print('output_layer 6: ', output_layer6) 
        output_layer5 = decoder_block(output_layer6, input_layer4, filters*16) 
        print('output_layer 5: ', output_layer5) 
        output_layer4 = decoder_block(output_layer5, input_layer3, filters*8)  
        print('output_layer 4: ', output_layer4) 
        output_layer3 = decoder_block(output_layer4, input_layer2, filters*4)  
        print('output_layer 3: ', output_layer3) 
        output_layer2 = decoder_block(output_layer3, input_layer1, filters*2) 
        print('output_layer 2: ', output_layer2) 
        output_layer1 = decoder_block(output_layer2, None, 3) 
        print('output_layer 1: ', output_layer1) 
    
        # The function returns the output layer of your model. "output_layer 1" is the final layer obtained from the last decoder_block()
        return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(output_layer1)


## Training

The following cells will use the FCN you created and define an ouput layer based on the size of the processed image and the number of classes recognized. The hyperparameters will be defined to compile and train the model.


    image_hw = 256
    image_shape = (image_hw, image_hw, 3)
    inputs = layers.Input(image_shape)
    num_classes = 3

    #Call fcn_model()
    output_layer = fcn_model(inputs, num_classes)


#### List shape for each layer in the model:
```
inputs :  Tensor("input_13:0", shape=(?, 256, 256, 3), dtype=float32)
input_layer 1 : Tensor("batch_normalization_111/batchnorm/add_1:0", shape=(?, 128, 128, 32), dtype=float32)
input_layer 2 : Tensor("batch_normalization_112/batchnorm/add_1:0", shape=(?, 64, 64, 64), dtype=float32)
input_layer 3 : Tensor("batch_normalization_113/batchnorm/add_1:0", shape=(?, 32, 32, 128), dtype=float32)
input_layer 4 : Tensor("batch_normalization_114/batchnorm/add_1:0", shape=(?, 16, 16, 256), dtype=float32)
input_layer 5 : Tensor("batch_normalization_115/batchnorm/add_1:0", shape=(?, 8, 8, 512), dtype=float32)
input_layer 6 : Tensor("batch_normalization_116/batchnorm/add_1:0", shape=(?, 4, 4, 1024), dtype=float32)
small_ip_layer : Tensor("batch_normalization_117/batchnorm/add_1:0", shape=(?, 4, 4, 2048), dtype=float32)
output_layer 6:  Tensor("batch_normalization_118/batchnorm/add_1:0", shape=(?, 8, 8, 1024), dtype=float32)
output_layer 5:  Tensor("batch_normalization_119/batchnorm/add_1:0", shape=(?, 16, 16, 512), dtype=float32)
output_layer 4:  Tensor("batch_normalization_120/batchnorm/add_1:0", shape=(?, 32, 32, 256), dtype=float32)
output_layer 3:  Tensor("batch_normalization_121/batchnorm/add_1:0", shape=(?, 64, 64, 128), dtype=float32)
output_layer 2:  Tensor("batch_normalization_122/batchnorm/add_1:0", shape=(?, 128, 128, 64), dtype=float32)
output_layer 1:  Tensor("batch_normalization_123/batchnorm/add_1:0", shape=(?, 256, 256, 3), dtype=float32)
```
### Hyperparameters

Define and tune the hyperparameters.
 ** learning_rate: ** learning_rate is set as 0.001, then try 0.0005 and 0.0002 after loss value or final score no improve for several steps.
 ** batch_size: ** number of training samples/images that get propagated through the network in a single pass.
 ** num_epochs: ** number of times the entire training dataset gets propagated through the network.
 ** steps_per_epoch: ** number of batches of training images that go through the network in 1 epoch. 
     One recommended value to try would be based on the total number of images in training dataset divided by the batch_size.
 ** validation_steps: ** number of batches of validation images that go through the network in 1 epoch. 
     This is similar to steps_per_epoch, except validation_steps is for the validation dataset.     
 ** workers: ** maximum number of processes to spin up. This can affect the training speed and is dependent on the hardware.


    learning_rate = 0.001
    batch_size = 32
    num_epochs = 80
    steps_per_epoch = 200
    validation_steps = 50
    workers = 2
  
## Training code

    import time

    # Define the Keras model and compile it for training
    model = models.Model(inputs=inputs, outputs=output_layer)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='categorical_crossentropy')

    t1 = time.time()

    # Data iterators for loading the training and validation data
    train_iter = data_iterator.BatchIteratorSimple(batch_size=batch_size,
                                                    data_folder=os.path.join('..', 'data', 'train'),
                                               image_shape=image_shape,
                                               shift_aug=True)

    val_iter = data_iterator.BatchIteratorSimple(batch_size=batch_size,
                                             data_folder=os.path.join('..', 'data', 'validation'),
                                             image_shape=image_shape)

    logger_cb = plotting_tools.LoggerPlotter()
    callbacks = [logger_cb]

    history = model.fit_generator(train_iter,
                    steps_per_epoch = steps_per_epoch, # the number of batches per epoch,
                    epochs = num_epochs, # the number of epochs to train for,
                    validation_data = val_iter, # validation iterator
                    validation_steps = validation_steps, # the number of batches to validate on
                    callbacks=callbacks,
                    workers = workers)
    print(history)

    t2 = time.time()
    print("Time: %0.2fs" % (t2 - t1))


## Run training code
The python training code can be found in ./code/my_model_training.py

[image_2]: ./docs/misc/training_flow.png 
![alt text][image_2] 

[image_3]: ./docs/misc/training_curves.png 
![alt text][image_3] 

# Prediction 
The predictions is based on the validation dataset. These predictions can be compared to the mask images, which are the ground truth labels, to evaluate how well the model is doing under different conditions.
There are three different predictions available from the helper code provided:
    1.patrol_with_targ: Test how well the network can detect the hero from a distance.
    2.patrol_non_targ: Test how often the network makes a mistake and identifies the wrong person as the target.
    3.following_images: Test how well the network can identify the target while following them.

The following cell will write predictions to files and return paths to the appropriate directories. The run_num parameter is used to define or group all the data for a particular model run. 


    run_num = 'run_3'

    val_with_targ, pred_with_targ = model_tools.write_predictions_grade_set(model,
                                        run_num,'patrol_with_targ', 'sample_evaluation_data') 

    val_no_targ, pred_no_targ = model_tools.write_predictions_grade_set(model, 
                                        run_num,'patrol_non_targ', 'sample_evaluation_data') 

    val_following, pred_following = model_tools.write_predictions_grade_set(model,
                                        run_num,'following_images', 'sample_evaluation_data')

# Result evaluation
Evaluate the model! The following cells include several different scores to help evaluating the model under the different conditions discussed during the Prediction step.

## Scores for while the quad is following behind the target. 
    true_pos1, false_pos1, false_neg1, iou1 = scoring_utils.score_run_iou(val_following, pred_following)

 1. number of validation samples intersection over the union evaulated on 542
 2. average intersection over union for background is 0.9966299392160006
 3. average intersection over union for other people is 0.4262138032784614
 4. average intersection over union for the hero is 0.941056498362849
 5. number true positives: 539, number false positives: 0, number false negatives: 0


## Scores for images while the quad is on patrol and the target is not visable
    true_pos2, false_pos2, false_neg2, iou2 = scoring_utils.score_run_iou(val_no_targ, pred_no_targ)

 1. number of validation samples intersection over the union evaulated on 270
 2. average intersection over union for background is 0.990500069409552
 3. average intersection over union for other people is 0.8154973057626096
 4. average intersection over union for the hero is 0.0
 5. number true positives: 0, number false positives: 31, number false negatives: 0

## This score measures how well the neural network can detect the target from far away
    true_pos3, false_pos3, false_neg3, iou3 = scoring_utils.score_run_iou(val_with_targ, pred_with_targ)

 1. number of validation samples intersection over the union evaulated on 322
 2. average intersection over union for background is 0.997071738369288
 3. average intersection over union for other people is 0.48694272888643103
 4. average intersection over union for the hero is 0.2420349581819363
 5. number true positives: 134, number false positives: 0, number false negatives: 178

-------------------------------------------------------------------------------------------------

    # Sum all the true positives, etc from the three datasets to get a weight for the score
    true_pos = true_pos1 + true_pos2 + true_pos3
    false_pos = false_pos1 + false_pos2 + false_pos3
    false_neg = false_neg1 + false_neg2 + false_neg3

    weight = true_pos/(true_pos+false_neg+false_pos)
    print('weight =' , weight)

weight = 0.763039

    # The IoU for the dataset that never includes the hero is excluded from grading
    final_IoU = (iou1 + iou3)/2
    print('final_IoU =  ', final_IoU)

final_IoU =   0.5915457282723926

    # And the final grade score is 
    final_score = final_IoU * weight
    print('final_score = ',final_score)

final_score =  0.45137219402190504

# Experimentation and testing
1. The loss value should be lower than 0.01, the code starts to call prediction function
2. For final score is great .40, the code starts to call evaluation function
3. It is not always true that lower loss value related higher final score
4. Add more train data can improve the final score
5. Using learning rate 0.001 to start training after several steps if loss is not lower or final score is not higher, exit current epoch loop, go into lower learning rate 0.0005, then 0.0002.
6. Save model to a file when higher final score is reached every time.
7. The best learning rate is 0.0005 for this training model

# Data set in testing
Use three data sets in the training
### 1. Original data from project
    found 2701 files in ../data/original/train/masks/
    no hero (pixel <= 3), file number = 2067
    hero very far (pixel > 3 and <= 65), file number = 47 
    hero not too close (pixel > 65 and <= 655), file number = 188
    hero close (pixel > 655), file number = 399

### 2. Record more data from Simulator
    found 6249 files in ../data/train/masks/
    no hero (pixel <= 3), file number = 4569
    hero very far (pixel > 3 and <= 65), file number = 979 
    hero not too close (pixel > 65 and <= 655), file number = 299
    hero close (pixel > 655), file number = 402

### 3. Use utility code to filter and flip the current files to generate more data
    found 7555 files in ../data/test1/train/masks/
    no hero (pixel <= 3), file number = 4590
    hero very far (pixel > 3 and <= 65), file number = 1958 
    hero not too close (pixel > 65 and <= 655), file number = 598
    hero close (pixel > 655), file number = 409


# Future Enhancements
1. Use different Optimizer
2. Record more data samples and use data augmentation to create a larger dataset
3. Use GAN as training model
4. Use database to store hyperparameters and testing results to help finding best results



