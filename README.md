# 基于Basicsr的低光增强代码
---
## R2-MWCNN
代码基于文章[LOW-LIGHT IMAGE ENHANCEMENT IN THE FREQUENCY DOMAIN](https://arxiv.org/pdf/2306.16782.pdf)的方法  
此处为基于tensorflow的[源代码](https://github.com/chqwer2/Low-light-Enhancement-in-the-Frequency-Domain)  
  
**实验结果**
在experiments文件夹中记录着参数设置和训练日志  
训练曲线如下所示：
* Enhancement_R2MWCNN_LOL_V1_2
  ![Enhancement_R2MWCNN_LOL_V1_2](https://github.com/lajiman/Lowlight-Enhancement-Experiment-on-Basicsr/blob/master/imgs/Enhancement_R2MWCNN_LOL_V1_2.png)
* Enhancement_R2MWCNN_VIT_LOL_V1_2(loss权重：0.0001)
  ![Enhancement_R2MWCNN_VIT_LOL_V1_2](https://github.com/lajiman/Lowlight-Enhancement-Experiment-on-Basicsr/blob/master/imgs/Enhancement_R2MWCNN_VIT_LOL_V1_2.png)
* Enhancement_R2MWCNN_VIT_LOL_V1_3(loss权重：0.01)
  ![Enhancement_R2MWCNN_VIT_LOL_V1_3](https://github.com/lajiman/Lowlight-Enhancement-Experiment-on-Basicsr/blob/master/imgs/Enhancement_R2MWCNN_VIT_LOL_V1_3.png)
