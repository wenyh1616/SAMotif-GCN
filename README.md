## Introduction
This repository holds the codebase for the paper:

**Motif-GCNs with Local and Non-Local Temporal Blocks for Skeleton-Based Action Recognition** Yu-Hui Wen, Lin Gao, Hongbo Fu, Fang-Lue Zhang, Shihong Xia, Yong-Jin Liu, TPAMI 2022. [[Early Access]](https://ieeexplore.ieee.org/document/9763364)

And this branch is the jittor version of the project


## Prerequisites
- Python3 (>3.5)
- jittor(just run `pip install jittor`)
- installed by `pip install -r requirements.txt`


### Installation
``` shell
git clone -b jittor https://github.com/wenyh1616/SAMotif-GCN.git 
cd SAMotif-GCN
cd torchlight; python setup.py install; cd ..
```

# Data Preparation

 - Download the raw data from [NTU-RGB+D](https://github.com/shahroudy/NTURGB-D) and [Skeleton-Kinetics](https://github.com/yysijie/st-gcn). 
            

[https://github.com/shahroudy/NTURGB-D]: NTU-RGB+D
[https://github.com/yysijie/st-gcn]: Skeleton-Kinetics

 - Preprocess the data with
  
    `python tools/ntu_gendata.py`
    
    `python tools/kinetics-gendata.py.`

 - Generate the bone data with: 
    
    `python tools/gen_bone_data.py`
    
    
## Training
To train a new model, run

```
python main.py recognition -c config/st_gcn/<dataset>/train.yaml
```
where the ```<dataset>``` can be ```ntu-xsub```, ```ntu-xview``` or ```kinetics```,  depending on the dataset you want to use.
The training results, including **model weights**, configurations and logging files, will be saved under the ```./work_dir``` by default or ```<work folder>``` if you appoint it.

You can modify the training parameters such as ```work_dir```, ```batch_size```, ```step```, ```base_lr``` and ```device``` in the command line or configuration files. The order of priority is:  command line > config file > default parameter. 

if you want to train it on multiple gpus,run

```
CUDA_VISIBLE_DEVICES="0,1" mpirun -np 2 python main_determine_sparse_intri.py recognition -c config/st_gcn/<dataset>/train.yaml
```

You can mofify the gpus in the CUDA_VISIBLE_DEVICES parameter. The "-np" parameter refers to the number of the gpus you want to train on.


Finally, custom model evaluation can be achieved by this command as we mentioned above:
```
python main.py recognition -c config/st_gcn/<dataset>/test.yaml --weights <path to model weights>
```

You can also run

```
CUDA_VISIBLE_DEVICES="0,1" mpirun -np 2 python main_determine_sparse_intri.py recognition -c config/st_gcn/<dataset>/test.yaml --weights <path to model weights>
```

to test model on multiple gpus.

## Citation

Please cite the following paper if you use this repository in your reseach.
```
@article{wen2022motif,
  title={Motif-GCNs with Local and Non-Local Temporal Blocks for Skeleton-Based Action Recognition},
  author={Wen, Yu-Hui and Gao, Lin and Fu, Hongbo and Zhang, Fang-Lue and Xia, Shihong and Liu, Yong-Jin},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2022},
  publisher={IEEE}
}
```

## Contact
For any question, feel free to contact
```
Yu-Hui Wen: wenyh1616@gmail.com
```

## Special thanks

The project is translated from pytorch version to jittor version by ChangSong Lei. If you have any quesion about the implementation in jittor version,feel free to contact

```
leics19@mails.tsinghua.edu.cn
```

