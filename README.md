# MaskFusion

    他们在SLAM技术的基础上引入了实例语义分割，效果视频相当惊艳，一起来看看吧。

    作者演示并声称该RGB-D SLAM系统不仅能实现实时的场景感知3D重建，更是具有吸引人的三大特点：

      1.实例感知。无需事先给定物体的先验知识或者已知模型，也能进行场景中的多目标识别；

      2.语义分割。借助于语义实例分割技术，能够实时在场景中对物体分配语义标签；

      3.动态追踪。尽管场景中的物体相互位置有不断变化，仍能实时分割、重建、语义标注。

    这个项目库包含MaskFusion，
    一个实时、对象感知、语义和动态的RGB-D SLAM系统，
    它超越了传统的只输出几何地图的系统——MaskFusion在跟踪和记录时识别、分割场景中的不同对象并为其分配语义类标签。
    即使当它们独立于摄像机移动时也会构造它们。

    当RGB-D相机扫描一个杂乱的场景时，基于图像的实例级语义分割创建了语义对象掩码，
    该语义对象掩码支持实时对象识别并为世界地图创建对象级表示。
    与以往的基于识别的SLAM系统不同，MaskFusion不需要先验知识或已知的对象模型，
    它可以识别并且可以处理多个独立的运动。与最近支持语义的SLAM系统不同，
    执行体素级语义分割的SLAM系统MaskFusion充分利用了使用实例级语义分割来使语义标签融合到对象感知映射中。
    我们展示了增强现实应用程序，演示了MaskFusion输出的地图的独特特性：实例感知、语义和动态。

[ 项目主页](http://visual.cs.ucl.ac.uk/pubs/maskfusion/index.html).

[![Figure of MaskFusion](figures/teaser.jpg "Click me to see a video.")](http://visual.cs.ucl.ac.uk/pubs/maskfusion/MaskFusion.mp4)

    本文提出的MaskFusion算法可以解决这两个问题，首先，可以从Object-level理解环境，
    在准确分割运动目标的同时，可以识别、检测、跟踪以及重建目标。
    
    分割算法由两部分组成：
     1. 2d语义分割： Mask RCNN:提供多达80类的目标识别等
     2. 利用Depth以及Surface Normal等信息向Mask RCNN提供更精确的目标边缘分割。
     
    上述算法的结果输入到本文的Dynamic SLAM框架中。
       使用Instance-aware semantic segmentation比使用pixel-level semantic segmentation更好。
       目标Mask更精确，并且可以把不同的object instance分配到同一object category。
     
    本文的作者又提到了现在SLAM所面临的另一个大问题：Dynamic的问题。
    作者提到，本文提出的算法在两个方面具有优势：
        相比于这些算法，本文的算法可以解决Dynamic Scene的问题。
        本文提出的算法具有Object-level Semantic的能力。
        
        
    所以总的来说，作者就是与那些Semantic Mapping的方法比Dynamic Scene的处理能力，
    与那些Dynamic Scene SLAM的方法比Semantic能力，在或者就是比速度。
    确实，前面的作者都只关注Static Scene， 现在看来，
    实际的SLAM中还需要解决Dynamic Scene(Moving Objects存在)的问题。}
    
![](https://github.com/Ewenwan/texs/blob/master/PaperReader/SemanticSLAM/MaskFusion0.png)
    
    每新来一帧数据，整个算法包括以下几个流程：

    1. 跟踪 Tracking
       每一个Object的6 DoF通过最小化一个能量函数来确定，这个能量函数由两部分组成：
          a. 几何的ICP Error;
          b. Photometric cost。
       此外，作者仅对那些Non-static Model进行Track。
       最后，作者比较了两种确定Object是否运动的方法：
          a. Based on Motioin Incosistency
          b. Treating objects which are being touched by a person as dynamic
          
    2. 分割 Segmentation
       使用了Mask RCNN和一个基于Depth Discontinuities and surface normals 的分割算法。
       前者有两个缺点：物体边界不精确、运行不实时。
       后者可以弥补这两个缺点， 但可能会Oversegment objects。
       
    3. 融合 Fusion
       就是把Object的几何结构与labels结合起来。

## 论文 Publication

* [MaskFusion: Real-Time Recognition, Tracking and Reconstruction of Multiple Moving Objects](https://arxiv.org/abs/1804.09194), Martin Rünz, Maud Buffier, Lourdes Agapito, ISMAR '18

## 编译 Building MaskFusion
    脚本 `build.sh` 展示了构建 MaskFusion的详细过程，以及相关的依赖项

### CMake options 编译选项
* `MASKFUSION_GPUS_MASKRCNN`: 目标分割使用的 GPU序列，与 SLAM 使用的 GPU分开
* `MASKFUSION_GPU_SLAM`:      SLAM使用的GPU序列， 主要使用者是 OpenGL 
* `MASKFUSION_MASK_RCNN_DIR`: MASKRCNN的安装路径 [Matterport MaskRCNN](https://github.com/matterport/Mask_RCNN)
* `MASKFUSION_NUM_GSURFELS`:  全局模型(环境模型)数量 Surfels allocated for environment model
* `MASKFUSION_NUM_OSURFELS`:  目标模型数量 Surfels allocated per object model
* `PYTHON_VE_PATH`:           PYTHON 虚拟环境路径 Path to (the root of) virtual python environment, used for tensorflow

### 依赖项 Dependencies
* Python3
* Tensorflow (>1.3.0, tested with 1.8.0)
* Keras (>2.1.2)
* MaskRCNN


## 运行 Running MaskFusion

* **Select the object categories** you would like to label by MaskRCNN. 

         调整识别的物体类别 `FILTER_CLASSES` within `Core/Segmentation/MaskRCNN/MaskRCNN.py.in`.
         For instance, `FILTER_CLASSES = ['person', 'skateboard', 'teddy bear']` 
         results in _skateboards_ 和 _teddy bears_ 被跟踪. 
         _person_ 被忽略. 
         空的数组，意味着所有的类别都会使用。

* 跟踪固定的目标

        enabled / disabled by calling `makeStatic()` and `makeNonStatic()` of instances of the `Model` class. 

        The overall system runs more robustly if objects are only tracked when being touched by a person. 

        We are **not** providing hand-detection software at the moment.

## Dataset and evaluation tools

### Tools
* 记录 Recorder for klg files: https://github.com/mp3guy/Logger2
* 可视化 Viewer for klg files: https://github.com/mp3guy/LogView
* 转换1  Images -> klg converter: https://github.com/martinruenz/dataset-tools/tree/master/convert_imagesToKlg
* 转换2  klg -> images/pointclouds: https://github.com/martinruenz/dataset-tools/tree/master/convert_klg
* 评估   Evaluate segmentation (intersection-over-union): https://github.com/martinruenz/dataset-tools/tree/master/evaluate_segmentation
* 合成数据集 Scripts to create synthetic datasets with blender: https://github.com/martinruenz/dataset-tools/tree/master/blender

## 硬件  Hardware
    GPU.
    We used an Nvidia TitanX for most experiments, 
    but also successfully tested MaskFusion on a laptop computer with an Nvidia GeForce™ GTX 960M. 

    GPU内存使用：

    If your GPU memory is limited, the `MASKFUSION_NUM_GSURFELS` and `MASKFUSION_NUM_OSURFELS` CMake options can help reduce the memory footprint per model (global/object, respectively).

    CPU.
    While the tracking stage of MaskFusion calls for a fast GPU, 
    the motion based segmentation performance depends on the CPU and accordingly, having a nice processor helps as well.

## 基于 ElasticFusion
The overall architecture and terminal-interface of MaskFusion is based on [ElasticFusion](https://github.com/mp3guy/ElasticFusion) and the ElasticFusion [readme file](https://github.com/mp3guy/ElasticFusion/blob/master/README.md) contains further useful information.


## 新的命令行参数 New command line parameters (see [source-file](https://github.com/martinruenz/maskfusion/blob/master/GUI/MainController.cpp#L34-L96))

* **-method:**        Method used for segmentation (cofusion, maskfusion)
* **-frameQ:**        Set size of frame-queue manually
* **-run**:           Run dataset immediately (otherwise start paused).
* **-static**:        Disable multi-model fusion.
* **-confO**:         Initial surfel confidence threshold for objects (default 0.01).
* **-confG**:         Initial surfel confidence threshold for scene (default 10.00).
* **-segMinNew**:     Min size of new object segments (relative to image size)
* **-segMaxNew**:     Max size of new object segments (relative to image size)
* **-offset**:        Offset between creating models
* **-keep**:          Keep all models (even bad, deactivated)
* **-dir**:           Processes a log-directory (Default: Color####.png + Depth####.exr [+ Mask####.png])
* **-depthdir**:      Separate depth directory (==dir if not provided)
* **-maskdir**:       Separate mask directory (==dir if not provided)
* **-exportdir**:     Export results to this directory, otherwise not exported
* **-basedir**:       Treat the above paths relative to this one (like depthdir = basedir + depthdir, default "")
* **-colorprefix**:   Specify prefix of color files (=="" or =="Color" if not provided)
* **-depthprefix**:   Specify prefix of depth files (=="" or =="Depth" if not provided)
* **-maskprefix**:    Specify prefix of mask files (=="" or =="Mask" if not provided)
* **-indexW**:        Number of digits of the indexes (==4 if not provided)
* **-nm**:            Ignore Mask####.png images as soon as the provided frame was reached.
* **-es**:            Export segmentation
* **-ev**:            Export viewport images
* **-el**:            Export label images
* **-em**:            Export models (point-cloud)
* **-en**:            Export normal images
* **-ep**:            Export poses after finishing run (just before quitting if '-q')
* **-or**:            Outlier rejection strength (default 3).

## 技巧 Tips

### 线下运行 Running MaskRCNN offline, before executing MaskFusion
You can use the script `Core/Segmentation/MaskRCNN/offline_runner.py` to extract masks readable by MaskFusion and visualisations. Use the `-maskdir` parameter to input these masks into MaskFusion.
Example usage: `./offline_runner.py -i /path/to/rgb/frames -o /path/to/output/masks --filter teddy_bear`

The visualization of the output will look like this:

![Figure MaskRCNN](figures/segmentation.jpg)

### Resolve the exception '***Could not open MaskRCNN module***':
* Check python output (run directly in terminal)
* Check value of CMake option `MASKFUSION_MASK_RCNN_DIR`
* Check value of CMake option `MASKFUSION_PYTHON_VE_PATH`
* Check if python package [pycocotools](https://github.com/waleedka/coco) is missing
* Check if python package [imgaug](https://github.com/aleju/imgaug) is missing
* Check if enough GPU memory is available
* Check variables `PYTHON_VE_PATH` and `MASK_RCNN_DIR` in `MaskRCNN.py` in your build directory

### Resolve the exception '***cudaSafeCall() Runtime API error : unknown error.***' at start-up
One reason for having this exception at start-up can be that OpenGL and Cuda are unable to share memory. Double-check the cmake parameter `MASKFUSION_GPU_SLAM`, especially in a multi-gpu setup.

### Using cv::imshow for debugging
`cv::imshow(...)` requires the library `libopencv_highgui.so`, which might (if GTK is used) depend on `libmirprotobuf.so` and hence on a specific *protobuf* version. The program, however, is also going to require a specific *protobuf* version and it can happen that the two versions are clashing leading to an error message like this: *This program requires version 3.5.0 of the Protocol Buffer runtime library, but the installed version is 2.6.1.  Please update your library.  If you compiled the program yourself, make sure that your headers are from the same version of Protocol Buffers as your link-time library.*
The easiest fix is to compile OpenCV with `-DWITH_QT=ON`, which removes the *protobuf* dependency of `libopencv_highgui.so`.

## License
MaskFusion includes the third-party open-source software ElasticFusion, which itself includes third-party open-source software. Each of these components have their own license.

You can find the ElasticFusion license in the file [LICENSE-ElasticFusion.txt](LICENSE-ElasticFusion.txt) and
the MaskFusion license in the file [LICENSE-MaskFusion.txt](LICENSE-MaskFusion.txt)
