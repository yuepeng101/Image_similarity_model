## Image_similarity_model
Implemented a feature-based similarity and retrieval model on a COCO-format dataset using PyTorch.

This repository presents a feature-based similarity and retrieval model implementation for a COCO-format dataset using PyTorch. The primary objective of this project is to create a model capable of retrieving similar images based on their features. To achieve this, we leverage a  pretrained ResNet-50 model to extract image features from the COCO dataset,  and then compute cosine similarities to retrieve similar images.. Before getting started, please ensure that you have the necessary libraries installed, including PyTorch, torchvision, and the COCO API.

### Implementation Overview
Here's a brief overview of what this code accomplishes:

- Data Loading: We load the COCO dataset, which serves as our image database.

- Feature Extraction: We utilize a pretrained ResNet-50 model to extract rich image features. These features capture the essential characteristics of each image.

- Similarity Computation: After feature extraction, we compute cosine similarities between images. This enables us to identify images that are most similar to a given query image.

- Image Retrieval: With the computed similarities, we create a retrieval mechanism to find and display images that are similar to a query image. This functionality allows users to explore visually related images within the dataset.

### COCO Dataset Download and Extraction Guide:

```python
# Downloading and Extracting the COCO Dataset
import wget  

!mkdir coco  
!cd ./coco  
!mkdir dataset  
!cd ./dataset  

wget.download("http://images.cocodataset.org/zips/val2017.zip", './')  
!unzip val2017.zip  
!rm val2017.zip  

!cd ../  
wget.download("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", './')  
!unzip annotations_trainval2017.zip  
!rm annotations_trainval2017.zip  
