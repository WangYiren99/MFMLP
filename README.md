# MFMLP
This is the codes of Multu-stage Fusion MLP for Compressive Spectral Imaging
## 1. Comparison with State-of-the-art Methods
<p align="center">
<img src="Images/comparison.png" width="600">
</p>
Fig. 1 PSNR-Params-FLOPs comparisons of our MFMLP and SOTA methods. The vertical axis is PSNR(dB), the horizontal axis is FLOPs(G), and the circle area is Params(M).
## 2. Architecture
<p align="center">
<img src="Images/architecture.png" width="900">
</p>
Fig. 2 The architecture of MFMLP with K stages(iterations).
<p align="center">
<img src="Images/MDDP.png" width="900">
</p>
Fig. 3 The overall architecture of MDDP. (a) MDDP adopts a U-shape structure. The green blocks are the input features of the kth stage, and also the output features of the (k − 1)th stage; the orange blocks are the output features of the kth stage, and also the input features of the (k + 1)th stage. (b)
SSMLP is composed of a Spatial Projection, a Spectral Projection, an input embedding, an output embedding, and two layer normalization. (c) Components
of Spectral Projection. (d) Components of Spatial Projection
## 3. Dataset
Download cave_1024_28 ([Baidu Disk](https://pan.baidu.com/s/1X_uXxgyO-mslnCTn4ioyNQ), code: `fo0q` | [One Drive](https://bupteducn-my.sharepoint.com/:f:/g/personal/mengziyi_bupt_edu_cn/EmNAsycFKNNNgHfV9Kib4osB7OD4OSu-Gu6Qnyy5PweG0A?e=5NrM6S)),
CAVE_512_28 ([Baidu Disk](https://pan.baidu.com/s/1ue26weBAbn61a7hyT9CDkg), code: `ixoe` | [One Drive](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/lin-j21_mails_tsinghua_edu_cn/EjhS1U_F7I1PjjjtjKNtUF8BJdsqZ6BSMag_grUfzsTABA?e=sOpwm4)),
KAIST_CVPR2021 ([Baidu Disk](https://pan.baidu.com/s/1LfPqGe0R_tuQjCXC_fALZA), code: `5mmn` | [One Drive](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/lin-j21_mails_tsinghua_edu_cn/EkA4B4GU8AdDu0ZkKXdewPwBd64adYGsMPB8PNCuYnpGlA?e=VFb3xP)), 
TSA_simu_data ([Baidu Disk](https://pan.baidu.com/s/1LI9tMaSprtxT8PiAG1oETA), code: `efu8` | [One Drive](https://1drv.ms/u/s!Au_cHqZBKiu2gYFDwE-7z1fzeWCRDA?e=ofvwrD)), 
TSA_real_data ([Baidu Disk](https://pan.baidu.com/s/1RoOb1CKsUPFu0r01tRi5Bg), code: `eaqe` | [One Drive](https://1drv.ms/u/s!Au_cHqZBKiu2gYFTpCwLdTi_eSw6ww?e=uiEToT)), and then put them into the corresponding folders of `datasets/` and recollect them as the following form:

```shell
|--MFMLP
    |--real——experiment
    |--simulation_experiment
    |--visualization
    |--datasets
        |--simulation_dataset
            |--cave_1024_28
                |--scene1.mat
                |--scene2.mat
                ：  
                |--scene205.mat
            |--KAIST_10
                |--scene01.mat
                |--scene02.mat
                ： 
                |--scene10.mat
            |--
        |--CAVE_512_28
            |--scene1.mat
            |--scene2.mat
            ：  
            |--scene30.mat
        |--KAIST_CVPR2021  
            |--1.mat
            |--2.mat
            ： 
            |--30.mat
        |--TSA_simu_data  
            |--mask.mat   
            |--Truth
                |--scene01.mat
                |--scene02.mat
                ： 
                |--scene10.mat
        |--TSA_real_data  
            |--mask.mat   
            |--Measurements
                |--scene1.mat
                |--scene2.mat
                ： 
                |--scene5.mat
```

Following TSA-Net and DGSMP, we use the CAVE dataset (cave_1024_28) as the simulation training set. Both the CAVE (CAVE_512_28) and KAIST (KAIST_CVPR2021) datasets are used as the real training set. 

## 4. Simulation Experiement

### 4.1　Training

```shell
cd MST/simulation/train_code/

# MST_S
python train.py --template mst_s --outf ./exp/mst_s/ --method mst_s 

```

The training log, trained model, and reconstrcuted HSI will be available in `MST/simulation/train_code/exp/` . 


### 4.2　Testing	

Download the pretrained model zoo from ([Google Drive](https://drive.google.com/drive/folders/1zgB7jHqTzY1bjCSzdX4lKQEGyK3bpWIx?usp=sharing) / [Baidu Disk](https://pan.baidu.com/s/1CH4uq_NZPpo5ra2tFzAdfQ?pwd=mst1), code: `mst1`) and place them to `MST/simulation/test_code/model_zoo/`

Run the following command to test the model on the simulation dataset.

```python
cd MST/simulation/test_code/

# MST_S
python test.py --template mst_s --outf ./exp/mst_s/ --method mst_s --pretrained_model_path ./model_zoo/mst/mst_s.pth

# MST_M
python test.py --template mst_m --outf ./exp/mst_m/ --method mst_m --pretrained_model_path ./model_zoo/mst/mst_m.pth

# MST_L
python test.py --template mst_l --outf ./exp/mst_l/ --method mst_l --pretrained_model_path ./model_zoo/mst/mst_l.pth

# CST_S
python test.py --template cst_s --outf ./exp/cst_s/ --method cst_s --pretrained_model_path ./model_zoo/cst/cst_s.pth

# CST_M
python test.py --template cst_m --outf ./exp/cst_m/ --method cst_m --pretrained_model_path ./model_zoo/cst/cst_m.pth

# CST_L
python test.py --template cst_l --outf ./exp/cst_l/ --method cst_l --pretrained_model_path ./model_zoo/cst/cst_l.pth

# CST_L_Plus
python test.py --template cst_l_plus --outf ./exp/cst_l_plus/ --method cst_l_plus --pretrained_model_path ./model_zoo/cst/cst_l_plus.pth

# GAP_Net
python test.py --template gap_net --outf ./exp/gap_net/ --method gap_net --pretrained_model_path ./model_zoo/gap_net/gap_net.pth

# ADMM_Net
python test.py --template admm_net --outf ./exp/admm_net/ --method admm_net --pretrained_model_path ./model_zoo/admm_net/admm_net.pth

# TSA_Net
python test.py --template tsa_net --outf ./exp/tsa_net/ --method tsa_net --pretrained_model_path ./model_zoo/tsa_net/tsa_net.pth

# HDNet
python test.py --template hdnet --outf ./exp/hdnet/ --method hdnet --pretrained_model_path ./model_zoo/hdnet/hdnet.pth

# DGSMP
python test.py --template dgsmp --outf ./exp/dgsmp/ --method dgsmp --pretrained_model_path ./model_zoo/dgsmp/dgsmp.pth

# BIRNAT
python test.py --template birnat --outf ./exp/birnat/ --method birnat --pretrained_model_path ./model_zoo/birnat/birnat.pth

# MST_Plus_Plus
python test.py --template mst_plus_plus --outf ./exp/mst_plus_plus/ --method mst_plus_plus --pretrained_model_path ./model_zoo/mst_plus_plus/mst_plus_plus.pth

# λ-Net
python test.py --template lambda_net --outf ./exp/lambda_net/ --method lambda_net --pretrained_model_path ./model_zoo/lambda_net/lambda_net.pth

# DAUHST-2stg
python test.py --template dauhst_2stg --outf ./exp/dauhst_2stg/ --method dauhst_2stg --pretrained_model_path ./model_zoo/dauhst_2stg/dauhst_2stg.pth

# DAUHST-3stg
python test.py --template dauhst_3stg --outf ./exp/dauhst_3stg/ --method dauhst_3stg --pretrained_model_path ./model_zoo/dauhst_3stg/dauhst_3stg.pth

# DAUHST-5stg
python test.py --template dauhst_5stg --outf ./exp/dauhst_5stg/ --method dauhst_5stg --pretrained_model_path ./model_zoo/dauhst_5stg/dauhst_5stg.pth

# DAUHST-9stg
python test.py --template dauhst_9stg --outf ./exp/dauhst_9stg/ --method dauhst_9stg --pretrained_model_path ./model_zoo/dauhst_9stg/dauhst_9stg.pth
```

- The reconstrcuted HSIs will be output into `MST/simulation/test_code/exp/`  

- Place the reconstructed results into `MST/simulation/test_code/Quality_Metrics/results` and  

```shell
Run cal_quality_assessment.m
```

to calculate the PSNR and SSIM of the reconstructed HSIs.

- #### Evaluating the Params and FLOPS of models

  We have provided a function `my_summary()` in `simulation/test_code/utils.py`, please use this function to evaluate the parameters and computational complexity of the models, especially the Transformers as 

```shell
from utils import my_summary
my_summary(MST(), 256, 256, 28, 1)
```

### 4.3　Visualization	

- Put the reconstruted HSI in `MST/visualization/simulation_results/results` and rename it as method.mat, e.g., mst_s.mat.

- Generate the RGB images of the reconstructed HSIs

```shell
 cd MST/visualization/
 Run show_simulation.m 
```

- Draw the spetral density lines

```shell
cd MST/visualization/
Run show_line.m
```

## 5. Real Experiement:

### 5.1　Training

```shell
cd MST/real/train_code/

# MST_S
python train.py --template mst_s --outf ./exp/mst_s/ --method mst_s 

# MST_M
python train.py --template mst_m --outf ./exp/mst_m/ --method mst_m  

# MST_L
python train.py --template mst_l --outf ./exp/mst_l/ --method mst_l 

# CST_S
python train.py --template cst_s --outf ./exp/cst_s/ --method cst_s 

# CST_M
python train.py --template cst_m --outf ./exp/cst_m/ --method cst_m  

# CST_L
python train.py --template cst_l --outf ./exp/cst_l/ --method cst_l

# CST_L_Plus
python train.py --template cst_l_plus --outf ./exp/cst_l_plus/ --method cst_l_plus

# GAP-Net
python train.py --template gap_net --outf ./exp/gap_net/ --method gap_net 

# ADMM-Net
python train.py --template admm_net --outf ./exp/admm_net/ --method admm_net 

# TSA-Net
python train.py --template tsa_net --outf ./exp/tsa_net/ --method tsa_net 

# HDNet
python train.py --template hdnet --outf ./exp/hdnet/ --method hdnet 

# DGSMP
python train.py --template dgsmp --outf ./exp/dgsmp/ --method dgsmp 

# BIRNAT
python train.py --template birnat --outf ./exp/birnat/ --method birnat 

# MST_Plus_Plus
python train.py --template mst_plus_plus --outf ./exp/mst_plus_plus/ --method mst_plus_plus 

# λ-Net
python train.py --template lambda_net --outf ./exp/lambda_net/ --method lambda_net

# DAUHST-2stg
python train.py --template dauhst_2stg --outf ./exp/dauhst_2stg/ --method dauhst_2stg

# DAUHST-3stg
python train.py --template dauhst_3stg --outf ./exp/dauhst_3stg/ --method dauhst_3stg

# DAUHST-5stg
python train.py --template dauhst_5stg --outf ./exp/dauhst_5stg/ --method dauhst_5stg

# DAUHST-9stg
python train.py --template dauhst_9stg --outf ./exp/dauhst_9stg/ --method dauhst_9stg
```

The training log, trained model, and reconstrcuted HSI will be available in `MST/real/train_code/exp/`


### 5.2　Testing	

```python
cd MST/real/test_code/

# MST_S
python test.py --template mst_s --outf ./exp/mst_s/ --method mst_s --pretrained_model_path ./model_zoo/mst/mst_s.pth

# MST_M
python test.py --template mst_m --outf ./exp/mst_m/ --method mst_m --pretrained_model_path ./model_zoo/mst/mst_m.pth

# MST_L
python test.py --template mst_l --outf ./exp/mst_l/ --method mst_l --pretrained_model_path ./model_zoo/mst/mst_l.pth

# CST_S
python test.py --template cst_s --outf ./exp/cst_s/ --method cst_s --pretrained_model_path ./model_zoo/cst/cst_s.pth

# CST_M
python test.py --template cst_m --outf ./exp/cst_m/ --method cst_m --pretrained_model_path ./model_zoo/cst/cst_m.pth

# CST_L
python test.py --template cst_l --outf ./exp/cst_l/ --method cst_l --pretrained_model_path ./model_zoo/cst/cst_l.pth

# CST_L_Plus
python test.py --template cst_l_plus --outf ./exp/cst_l_plus/ --method cst_l_plus --pretrained_model_path ./model_zoo/cst/cst_l_plus.pth

# GAP_Net
python test.py --template gap_net --outf ./exp/gap_net/ --method gap_net --pretrained_model_path ./model_zoo/gap_net/gap_net.pth

# ADMM_Net
python test.py --template admm_net --outf ./exp/admm_net/ --method admm_net --pretrained_model_path ./model_zoo/admm_net/admm_net.pth

# TSA_Net
python test.py --template tsa_net --outf ./exp/tsa_net/ --method tsa_net --pretrained_model_path ./model_zoo/tsa_net/tsa_net.pth

# HDNet
python test.py --template hdnet --outf ./exp/hdnet/ --method hdnet --pretrained_model_path ./model_zoo/hdnet/hdnet.pth

# DGSMP
python test.py --template dgsmp --outf ./exp/dgsmp/ --method dgsmp --pretrained_model_path ./model_zoo/dgsmp/dgsmp.pth

# BIRNAT
python test.py --template birnat --outf ./exp/birnat/ --method birnat --pretrained_model_path ./model_zoo/birnat/birnat.pth

# MST_Plus_Plus
python test.py --template mst_plus_plus --outf ./exp/mst_plus_plus/ --method mst_plus_plus --pretrained_model_path ./model_zoo/mst_plus_plus/mst_plus_plus.pth

# λ-Net
python test.py --template lambda_net --outf ./exp/lambda_net/ --method lambda_net --pretrained_model_path ./model_zoo/lambda_net/lambda_net.pth

# DAUHST_2stg
python test.py --template dauhst --outf ./exp/dauhst_2stg/ --method dauhst_2stg --pretrained_model_path ./model_zoo/dauhst/dauhst_2stg.pth

# DAUHST_3stg
python test.py --template dauhst --outf ./exp/dauhst_3stg/ --method dauhst_3stg --pretrained_model_path ./model_zoo/dauhst/dauhst_3stg.pth

# DAUHST_5stg
python test.py --template dauhst --outf ./exp/dauhst_5stg/ --method dauhst_5stg --pretrained_model_path ./model_zoo/dauhst/dauhst_5stg.pth

# DAUHST_9stg
python test.py --template dauhst --outf ./exp/dauhst_9stg/ --method dauhst_9stg --pretrained_model_path ./model_zoo/dauhst/dauhst_9stg.pth
```

- The reconstrcuted HSI will be output into `MST/real/test_code/exp/`  


### 5.3　Visualization	

- Put the reconstruted HSI in `MST/visualization/real_results/results` and rename it as method.mat, e.g., mst_plus_plus.mat.

- Generate the RGB images of the reconstructed HSI

```shell
cd MST/visualization/
Run show_real.m
```

## 6. 
If this repo helps you, please consider citing our works:


```shell


# MST
```
