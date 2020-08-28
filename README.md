# mgn-baseline

## 一、环境说明

- Ubuntu  16.04.1
- CUDA  10.0.130
- CUDNN  7.6.0
- python  3.7.6
- torch  1.3.1
- torchvision  0.4.2
- pillow  7.1.2
- faiss-gpu  1.6.3
- yacs  0.1.8
- gdown  3.12.2

## 二、结构说明

- fastreid目录  搭建模型
- checkpoints目录  存放训练好的模型的权重
- data目录  将下载的数据集放入./data目录
- features目录  存放图片提取的feature
- result目录  存放结果
- baseline.py  mgn-baseline
- baseline2.py  mgn-baseline 和上面的差别是batchsize和learnrate不一样
- defaults.py  配置模型训练参数
- get_query_features.py  提取查询集图片特征
- get_gallery_features.py  提取gallery图片集特征
- test.py  测试模型，结果可在result目录下查看，人工观察对比
- submit.py  生成提交结果，可使用余弦相似度，欧式距离以及曼哈顿距离进行查询

## 三、使用说明

1. 把下载的数据放入data/目录

2. 训练mgn模型

   ```
   $ python baseline.py
   ```

3. 获得图片feature

   ```
   $ python get_gallery_features.py
   $ python get_query_features.py
   ```

4. 生成提交文件

   ```
   $ python submit.py
   ```