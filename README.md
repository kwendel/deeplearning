# Deeplearning
Image caption generation using an encoder-decoder neural network architecture. VGGNet intermediate predictions are used for encoding the image data and a Transformer is used to decode to a caption.

### Authors
- W. Diepeveen
- N. van der Laan
- D. van Tetering
- P. Verkooijen
- K. Wendel

### Requirements
**Code**  
Project requirements can be found in `requirements.txt` and can be installed with `pip install -r requirements.txt`

**Pretrained models**  
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

### Dataset -- Flickr8k
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


 