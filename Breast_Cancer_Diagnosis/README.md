# Breast_Cancer_Classification

The first step in identifying breast cancer requires inspection of mammogram scans to find existence of lesion and its pathology as Benign/Malignant. With a lot of women having high probability of breast cancer it will be really helpful to radiologists if we can help speed up this process of inspecting mammograms to find lesions and their nature. Use of deep learning methods to identify lesions can significantly help and support the current radiologists. We use various deep learning methods with residual connections and transfer learning approaches for this classification task. In addition to some very popularly known architectures, we've experimented with our very own custom deep CNN architecture which performs better than a few of them. We also show heatmaps with bounding box highlighting the Region of Interest ROI. 

This project aims to classify mammogram scans as Benign vs. Malignant using deep learning models such as ResNet18, ResNet34, and customized convolutional neural network. Given transfer learning method and a thorough hyper-parameter tuning process, ResNet18 showes best performance among all models. 

The project consists of multiple phases: 
1. Collecting and pre-processing images from the data set CBIS-DDSM
2. Developing models
3. Training model on sampled train set
4. Tuning hyper-parameters for each models on sampled validation set
5. Evaluating models on sampled test set 
6. Running models with best configuration for each model on the whole train / val / test sets 

This folder includes the folling matterials: 
1. excel_files: all excel files of names of images for sampled sets and full sets
2. graphs: graphs (AUC, loss, accuracy) from models run locally on a subset of 10 images
3. HPC_graphs: graphs (AUC, loss, accuracy) from models run on High performance computing cluster at NYU
4. HPC_Python_Scripts: scripts to run models on HPC python
5. images: 10 images to run models on local machine
6. Python_scripts: Python scripts for functions used to run models 
