
This is the official implementation of the approach described in the paper of PLACNet :

> [**PLACNet: Enhancing 3D Human Pose Estimation with a Pattern Library-Guided Attention-Convolution Dual-Stream Network**]
            
> Fengguang Hu<sup>1</sup>, Tuqian Zhang<sup>2</sup>, Yongkang Hu<sup>1</sup>, Xinrui Yu<sup>1</sup>,Shumei Bao<sup>1</sup>

> <sup>1</sup>School of Software, Xinjiang University,<sup>2</sup>School of Computer Science and Technology, Xinjiang University


## 💡 Environment
The project is developed under the following environment:
- Python 3.10.x
- PyTorch 2.2.1
- CUDA 12.1

For installation of the project dependencies, please run:
```
pip install -r requirements.txt
``` 

## 🐳 Dataset
### Human3.6M
#### Preprocessing
1. We follow the previous state-of-the-art method [MotionBERT](https://github.com/Walter0807/MotionBERT/blob/main/docs/pose3d.md) for dataset setup. Download the [MotionBERT](https://github.com/Walter0807/MotionBERT/blob/main/docs/pose3d.md)'s preprocessed H3.6M data [here](https://1drv.ms/u/s!AvAdh0LSjEOlgU7BuUZcyafu8kzc?e=vobkjZ) and unzip it to 'data/motion3d'.
2. Slice the motion clips by running the following python code in `data/preprocess` directory:

**For our model with T = 243**:
```text
python h36m.py  --n-frames 243
```
**or T = 81**
```text
python h36m.py  --n-frames 81
```
**or T = 27**
```text
python h36m.py  --n-frames 81
```


### MPI-INF-3DHP
#### Preprocessing
Please refer to [P-STMO](https://github.com/paTRICK-swk/P-STMO#mpi-inf-3dhp) for dataset setup. After preprocessing, the generated .npz files (`data_train_3dhp.npz` and `data_test_3dhp.npz`) should be located at `data/motion3d` directory.

## ✨ Training
After dataset preparation, you can train the model as follows:
### Human3.6M
You can train Human3.6M with the following command:
```
python train_PLACNet.py --config <PATH-TO-CONFIG>
```
where config files are located at `configs/h36m`. 
```
python train_PLACNet.py --config configs/h36m/PLACNet_h36m_243.yaml 
```
### MPI-INF-3DHP
You can train MPI-INF-3DHP with the following command:
```
python train_3dhp.py --config <PATH-TO-CONFIG>
```
where config files are located at `configs/mpi`. 
```
python train_3dhp.py --config configs/mpi/PLACNet_mpi_81.yaml 
```
### Human3.6M (GT)

Please refer to [MotionBERT](https://github.com/Walter0807/MotionBERT/blob/main/docs/pose3d.md).

## 🚅 Evaluation
|Human3.6M|243|[download](链接：https://pan.quark.cn/s/e340232bc8b9  提取码：TpsB)|


After downloading the weight from table above, you can evaluate Human3.6M models by:
```
python train_PLACNet.py --eval-only --checkpoint <CHECKPOINT-DIRECTORY> --checkpoint-file <CHECKPOINT-FILE-NAME> --config <PATH-TO-CONFIG>
```
For example if PLACNet with T = 243 of H.36M is downloaded and put in `checkpoint` directory, then you can run:
```
python train_PLACNet.py --eval-only  --checkpoint checkpoint --checkpoint-file best_epoch_PLACNet.pth.tr --config configs/h36m/PLACNet_h36m_243.yaml
```

For MPI-INF-3DHP dataset, you can download the checkpoint with T = 81 and put in `checkpoint_mpi` directory, then you can run:
```
python train_3dhp.py --eval-only  --checkpoint checkpoint_mpi --checkpoint-file PLACNet_mpi_81.pth.tr --config configs/mpi/PLACNet_mpi_81.yaml
```

## 👀 Visualization

For the 3D human pose estimation visualization, please refer to [MHFormer](https://github.com/Vegetebird/MHFormer).

For the attention matrix visualization, this is just a 243x243 matrix, and you can easily visualize it. Let GPT/DeepSeek help you!



## ✏️ Citation

If you find our work useful in your research, please consider citing:

    @article{liu2025tcpformer,
        title={TCPFormer: Learning Temporal Correlation with Implicit Pose Proxy for 3D Human Pose Estimation},
        author={Liu, Jiajie and Liu, Mengyuan and Liu, Hong and Li, Wenhao},
        journal={arXiv preprint arXiv:2501.01770},
        year={2025}
    }



## 👍 Acknowledgement

Our code is extended from the following repositories. We thank the authors for releasing the codes. 

- [MixSTE](https://github.com/JinluZhang1126/MixSTE)
- [MHFormer](https://github.com/Vegetebird/MHFormer)
- [StridedTransformer-Pose3D](https://github.com/Vegetebird/StridedTransformer-Pose3D)
- [MotionBERT](https://github.com/Walter0807/MotionBERT)
- [MotionAGFormer](https://github.com/TaatiTeam/MotionAGFormer)
- [TCPFormer](https://github.com/AsukaCamellia/TCPFormer)

## 🔒 Licence

This project is licensed under the terms of the MIT license.



