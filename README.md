# AI6103_Project
This is the group project of course AI6103 - Fall2022.   
November 15, 2022

## 1. Introduction
We take SVHN dataset and ResNet-18 network, use Adam as our stochastic optimization algorithm, and experiment with several regulation techniques, ranging from initial learning rate, learning rate schedule, weight decay, data augmentation, dropout and label-smoothing.  

The `project_codebase.py` file contains all classes and functions used in our experiments, where the parameters are set to the situation that leads to the highest test accuracy.  
The `*.ipynb` and `*.png` files in folders are the process and result of corresponding individual experiment. This is FYI as a "proof of efforts".

## 2. Group arrangements and optimal choices
Group members: **Jiang Haofeng [C]**, Cao Yifei, Chen Zhelong, Wu Dongjun, Yang Shunping 

### Initial learning rate
* Initial lr = 0.1 -- Haofeng;  
* Initial lr = 0.01 -- Haofeng;  
* Initial lr = 0.001 -- Haofeng.  

Finally, Initial lr = 0.001 is chosen to be optimal.  

### Learning rate schedule
* Constant lr -- Yifei;  
* Cosine annealing lr -- Dongjun;  
* Step lr -- Zhelong;  
* Linear lr -- Shunping;  
* Exponential lr -- Haofeng.  

Finally, Cosine annealing lr is chosen to be optimal.  

### Weight decay
* Weight decay coefficient = 5e-4 -- Zhelong;  
* Weight decay coefficient = 1e-2 -- Haofeng;  

Finally, Weight decay coefficient = 5e-4 is chosen to be optimal.  

### Data augmentation
* AutoAugment -- Dongjun;  
* RandAugment -- Yifei;  
* AugMix -- Shunping.

Finally, AutoAugment is chosen to be optimal.  

### Dropout
* Dropout -- Zhelong.

Finally, non-Dropout is chosen to be optimal.  

### Label smoothing
* Label smoothing coefficient = 0.001 -- Dongjun.  
* Label smoothing coefficient = 0.01 -- Yifei.  
* Label smoothing coefficient = 0.1 -- Shunping.  

Finally, Label smoothing coefficient = 0.1 is chosen to be optimal.  

### Other logistic work
* Set environment -- Haofeng; 
* Group arrangment -- Haofeng;
* Report -- Haofeng;
* Video -- Haofeng.  
