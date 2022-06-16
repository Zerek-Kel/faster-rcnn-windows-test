# 课设简介

fork了[jwyang的项目](https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0)

使用faster rcnn预训练模型对voc测试数据集进行目标检测，总结实验原理

# 关键代码

见[github地址](https://github.com/Zerek-Kel/faster-rcnn-for-windows-test)或[实验原理](#jump1)部分

# 实验设置

## 运行环境

Pytorch 1.0

Python 3.6

cython 0.11.0

matplotlib 3.3.4

opencv-python 4.6.0

pillow 8.4.0

numpy 1.19.5

scipy 1.2.1

torch 1.7.1+cpu

torchvision 0.8.2+cpu

torch 和 torchvision 可下载whl文件完成安装（faster-rcnn.pytorch-pytorch-1.0\whl）

## 数据下载

VOC2007数据集（faster-rcnn.pytorch-pytorch-1.0\data\VOCdevkit2007）

预训练模型权重文件（D:\faster-rcnn.pytorch-pytorch-1.0\models\res101\pascal_voc\faster_rcnn_1_20_2504.pth）

## 数据集介绍

使用VOC数据集

> VOC2007
>
> - Annotations
> - ImageSets
> - JPEGImages

Annotations存放标注，具体内容位于xml文件内的object

ImageSets下划分训练集、验证集和测试集

JPEGImages存放图片

## 测试参数设置

运行test_net.py

```shell
python test_net.py --dataset pascal_voc --net res101 --checksession 1 --checkepoch 20 --checkpoint 2504
```

--dataset 数据集 pascal_voc

--net 网络模型 res101

the specific model session, chechepoch and checkpoint  --checksession 1 --checkepoch 20 --checkpoint 2504

运行demo.py

```shell
python demo.py --net res101 --checksession 1 --checkepoch 20 --checkpoint 2504 --load_dir models --webcam_num 0
```

--load_dir 从models目录下读取模型权重文件

--webcam_num 使用本地摄像头读取图片

# <span id='jump1'>实验原理</span>

## 模型概述

![2](https://github.com/Zerek-Kel/faster-rcnn-windows-test/blob/master/md_image/2.jpg)

### 整体架构

1. Conv layers。作为一种CNN网络目标检测方法，Faster RCNN首先使用一组基础的conv+relu+pooling层提取image的feature maps。该feature maps被共享用于后续RPN层和全连接层。

2. Region Proposal Networks。RPN网络用于生成region proposals。该层通过softmax判断anchors属于positive或者negative，再利用bounding box regression修正anchors获得精确的proposals。

3. Roi Pooling。该层收集输入的feature maps和proposals，综合这些信息后提取proposal feature maps，送入后续全连接层判定目标类别。

4. Classification。利用proposal feature maps计算proposal的类别，同时再次bounding box regression获得检测框最终的精确位置。

   ![](https://github.com/Zerek-Kel/faster-rcnn-windows-test/blob/master/md_image/1.jpg)

Faster RCNN物体检测模型由三个模块组成：

- 特征提取网络    图片（img）经过预训练的网络（Extractor），提取到了图片的特征（feature）。
- RPN网络    利用提取的特征（feature），经过RPN网络，找出一定数量的rois（region of interests）。
- 区域归一化、物体分类以及边框回归    将rois和图像特征features，输入到RoIHead，对这些rois进行分类，判断都属于什么类别，同时对这些rois的位置进行微调。

### RPN

经典的检测方法生成检测框都非常耗时，如OpenCV adaboost使用滑动窗口+图像金字塔生成检测框；或如R-CNN使用SS(Selective Search)方法生成检测框。而Faster RCNN则抛弃了传统的滑动窗口和SS方法，直接使用RPN生成检测框，这也是Faster R-CNN的巨大优势，能极大提升检测框的生成速度。

![3](https://github.com/Zerek-Kel/faster-rcnn-windows-test/blob/master/md_image/3.jpg)

上图展示了RPN网络的具体结构。可以看到RPN网络实际分为2条线，上面一条通过softmax分类anchors获得positive和negative分类，下面一条用于计算对于anchors的bounding box regression偏移量，以获得精确的proposal。而最后的Proposal层则负责综合positive anchors和对应bounding box regression偏移量获取proposals，同时剔除太小和超出边界的proposals。其实整个网络到了Proposal Layer这里，就完成了相当于目标定位的功能。

### Anchor

在RPN中，作者提出了anchor。Anchor是大小和尺寸固定的候选框。论文中用到的anchor有三种尺寸和三种比例，如下图所示，三种尺寸分别是小（蓝128）中（红256）大（绿512），三个比例分别是1:1，1:2，2:1。3×3的组合总共有9种anchor。

![4](https://github.com/Zerek-Kel/faster-rcnn-windows-test/blob/master/md_image/4.png)

然后用这9种anchor在特征图（feature）左右上下移动，每一个特征图上的点都有9个anchor，最终生成了 (H/16)× (W/16)×9个anchor. 对于一个512×62×37的feature map，有 62×37×9~ 20000个anchor。softmax分类器提取positvie anchors -> bbox reg回归positive anchors -> Proposal Layer生成proposals

### ROI pooling

RoI Pooling层则负责收集proposal，并计算出proposal feature maps，送入后续网络。

- 由于proposal是对应MxN尺度的，所以首先使用spatial_scale参数将其映射回(M/16)x(N/16)大小的feature map尺度；
- 再将每个proposal对应的feature map区域水平分为 ![[公式]](https://github.com/Zerek-Kel/faster-rcnn-windows-test/blob/master/md_image/7.png)的网格；
- 对网格的每一份都进行max pooling处理。
- ![5](https://github.com/Zerek-Kel/faster-rcnn-windows-test/blob/master/md_image/5.jpg)

### Classification

Classification部分利用已经获得的proposal feature maps，通过full connect层与softmax计算每个proposal具体属于那个类别（如人，车，电视等），输出cls_prob概率向量；同时再次利用bounding box regression获得每个proposal的位置偏移量bbox_pred，用于回归更加精确的目标检测框。Classification部分网络结构如下图。
![6](https://github.com/Zerek-Kel/faster-rcnn-windows-test/blob/master/md_image/6.jpg)

# 评估

样例展示：

原图

![img1](https://github.com/Zerek-Kel/faster-rcnn-windows-test/blob/master/images/img1.jpg)

结果图：

![img1_det_res101](https://github.com/Zerek-Kel/faster-rcnn-windows-test/blob/master/images/img1_det_res101.jpg)

各类样本ap指标和map指标结果如下：

AP for aeroplane = 0.7079

AP for bicycle = 0.7954

AP for bird = 0.7666

AP for boat = 0.6223

AP for bottle = 0.5759

AP for bus = 0.7955

AP for car = 0.8337

AP for cat = 0.8861

AP for chair = 0.4950

AP for cow = 0.8077

AP for diningtable = 0.5676

AP for dog = 0.8592

AP for horse = 0.8394

AP for motorbike = 0.7834

AP for person = 0.7786

AP for pottedplant = 0.4456

AP for sheep = 0.7474

AP for sofa = 0.7460

AP for train = 0.7531

AP for tvmonitor = 0.7402

Mean AP = 0.7273

# 对实验结果的原理性分析

Faster RCNN摒弃了传统的滑动窗口和选择性搜索方法，直接使用RPN生成检测框，这也是Faster RCNN的巨大优势，能极大提升检测框的生成速度。

具体优点是：①提高了检测精度和速度；②真正实现了端到端的目标检测框架；③生成建议框仅需10ms。

具体缺点是：①还是无法达到实时检测目标；②获取Region Proposal，再对每个Proposal分类，计算量还是比较大。

# 结论与总结

VOC数据集上yolov4和Faster RCNN模型结果比较：

|                    | yolov4 | Faster RCNN |
| ------------------ | ------ | ----------- |
| AP for aeroplane   | 0.9618 | 0.7079      |
| AP for bicycle     | 0.9273 | 0.7954      |
| AP for bird        | 0.9094 | 0.7666      |
| AP for boat        | 0.8328 | 0.6223      |
| AP for bottle      | 0.8519 | 0.5759      |
| AP for bus         | 0.9472 | 0.7955      |
| AP for car         | 0.9648 | 0.8337      |
| AP for cat         | 0.9385 | 0.8861      |
| AP for chair       | 0.7835 | 0.4950      |
| AP for cow         | 0.9442 | 0.8077      |
| AP for diningtable | 0.8421 | 0.5676      |
| AP for dog         | 0.9263 | 0.8592      |
| AP for horse       | 0.9514 | 0.8394      |
| AP for motorbike   | 0.9505 | 0.7834      |
| AP for person      | 0.9288 | 0.7786      |
| AP for pottedplant | 0.6765 | 0.4456      |
| AP for sheep       | 0.9230 | 0.7474      |
| AP for sofa        | 0.8469 | 0.7460      |
| AP for train       | 0.9560 | 0.7531      |
| AP for tvmonitor   | 0.9135 | 0.7402      |
| mAP                | 0.8988 | 0.7273      |

AP方面yolov4和Faster RCNN表现一致，大多数类别都表现出很好的性能，但是在类别chair和pottedplant上一致表现较差，Faster RCNN在boat和bottle上也有很差的表现。

Faster RCNN是一个two stage的目标检测算法，把检测问题分成了两个阶段，第一个阶段是生成候选区域，第二个阶段是对候选区域位置进行调整以及分类，可解释性更好，错误识别率低，但是速度很慢。

one stage的目标检测算法yolov4的话速度相比于faster-rcnn来说就要快很多了，特征提取层采用了特征金字塔+下采样的结构以及训练时采用Mosaic数据增强的方法，所以对于小目标检测也有着不错的效果。
