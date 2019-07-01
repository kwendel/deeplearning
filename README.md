# Deeplearning: Attention-based image to caption using a Transformer based network
Image caption generation using an encoder-decoder neural network architecture. VGGNet intermediate predictions are used for encoding the image data to a hidden state. A Transformer network is used to decode to a caption.

### Authors
- W. Diepeveen
- N. van der Laan
- D. van Tetering
- P. Verkooijen
- K. Wendel

This codebase is an extension of the Transformer network from [https://github.com/kyubyong/transformer](https://github.com/kyubyong/transformer)

:clipboard: Requirements
------
#### Code  
Project requirements can be found in `requirements.txt` and can be installed with
```
pip install -r requirements.txt
```

#### Pretrained models
Preprocessing uses a pretrained VGG16 and GloVe Word2Vec. Please download the pretrained models.
- VGG16: https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5
- GloVe: http://nlp.stanford.edu/data/glove.6B.zip 
```
- project root
│
└---models
    └---pretrained
        |--- vgg16_weights_tf_dim_ordering_tf_kernels.h5
        └--- glove.6B
            └--- glove.6B.50d.txt
``` 

#### Dataset 
The Flicker8k dataset is used in this project and can be downloaded online (or contact the authors for a torrent).
Dataset must be provided in the following way (which is the default structure after downloading) and each file below must be defined:
```
- project root
│
└---dataset
    └---Flickr8k
        └--- Flickr8k_Dataset
            └--- Flicker8k_Dataset
                 |  *.jpg
        └--- Flickr8k_text            
             |  Flickr8k.lemma.token.txt
             |  Flickr_8k.devImages.txt
             |  Flickr_8k.testImages.txt
             |  Flickr_8k.trainImages.txt
``` 
:running: How to run
------

#### Hyperparameters
The network contains many hyperparameters which are set to sensible default values. For the full list, check out [hparams.py](utils/hparams.py)

Example usage: `python train.py --split_size 0.1 --logdir log/1`

#### :chart_with_upwards_trend: Network Usage
1. First, preprocess the Flickr8k dataset. The script will output a object from each created dataset split to show if the preprocessing was succesfull
    ```
    python prepro.py
    ```
2. Now we are ready for training the EncoderDecoder network with:
    ```
    python train.py
    ```
    The train script will shown in which epoch it currently is with the help of `tqdm`. A checkpoint of the model per epoch is automatically saved to the `--logdir (default=log/1)`, with the epoch number and current loss in the filename.

3. After training, the model can be evaluated on a separate testing data set with:
    ```
    python test.py --ckpt <checkpoint_path>
    ```

#### :mag: Analysing
During training, logs values will be created that can be analysed with the help of `tensorboard`. 
In each iteration the loss, learning rate and epoch will be logged. 
When the eval function is enabled during training, a random caption will be selected from the evaluation dataset and both the real and predicted caption will be logged to `tensorboard`

For running `tensorboard`:
```
tensorboard --logdir <log_directory>
```

#### :cloud: Google Cloud Console
For a guide on how to install the Google Cloud SDK for connection to the Google Cloud with `ssh`, check out [this guide](CLOUD.md).






 