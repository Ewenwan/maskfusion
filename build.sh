#!/bin/bash -e
#
# This is a build script for MaskFusion.
# 创建脚本
# 
# 参数 Use parameters:
# `--install-packages` ubuntu依赖项安装 to install required Ubuntu packages
# `--install-cuda`     安装显卡cuda加速 to install the NVIDIA CUDA suite
# `--build-dependencies` 安装第三方依赖库位置 to build third party dependencies
#
# Example:
#   ./build.sh --install-packages --build-dependencies
#
#   which will create:
#   - ./deps/densecrf
#   - ./deps/gSLICr
#   - ./deps/OpenNI2
#   - ./deps/Pangolin
#   - ./deps/opencv-3.1.0
#   - ./deps/boost (unless env BOOST_ROOT is defined)
#   - ./deps/coco
#   - ./deps/Mask_RCNN

# git clone 下载函数
# Function that executes the clone command given as $1 iff repo does not exist yet. Otherwise pulls.
# Only works if repository path ends with '.git'
# Example: git_clone "git clone --branch 3.4.1 --depth=1 https://github.com/opencv/opencv.git"
function git_clone(){
  repo_dir=`basename "$1" .git`
  git -C "$repo_dir" pull 2> /dev/null || eval "$1"
}

# 进入到文件夹
# Ensure that current directory is root of project
cd $(dirname `realpath $0`)

# bash中高亮显示字符
# Enable colors
source deps/bashcolors/bash_colors.sh
function highlight(){
  clr_magentab clr_bold clr_white "$1"
}

# ubuntu依赖项安装
if [[ $* == *--install-packages* ]] ; then
  # bash中高亮显示
  highlight "Installing system packages..."
  # Get ubuntu version:
  sudo apt-get install -y wget software-properties-common
  source /etc/lsb-release # fetch DISTRIB_CODENAME 获取DISTRIB_CODENAME  ubuntu版本
  
# Shell中的条件判断语句if []; ~ then ~fi===============================================
# if-then-elif-then-elif-then-...-else-fi。这种语句可以实现多重判断，注意最后一定要以一个else结尾。===========
# ubuntu 14.04  trusty
  if [[ $DISTRIB_CODENAME == *"trusty"* ]] ; then
    # g++ 4.9.4
    sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
    # cmake 3.2.2
    sudo add-apt-repository -y ppa:george-edison55/cmake-3.x
    # openjdk 8
    sudo add-apt-repository -y ppa:openjdk-r/ppa
  fi
  # 安装各种软件 输出重定向到呀设备 也就是不输出信息 > /dev/null 
  sudo apt-get update > /dev/null
  sudo apt-get install -y \
    build-essential \
    cmake \
    freeglut3-dev \
    g++-4.9 \
    gcc-4.9 \
    git \
    libeigen3-dev \
    libglew-dev \
    libjpeg-dev \
    libsuitesparse-dev \
    libudev-dev \
    libusb-1.0-0-dev \
    openjdk-8-jdk \
    unzip \
    zlib1g-dev \
    cython3
    
# virtualenv 是一个创建隔绝的Python环境的工具。========
    sudo -H pip3 install virtualenv

  if [[ $DISTRIB_CODENAME == *"trusty"* ]] ; then
     # switch to g++-4.9
     sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-4.9
     # switch to java-1.8.0
     sudo update-java-alternatives -s java-1.8.0-openjdk-amd64
  fi

fi # --install-packages

# 安装显卡cuda加速======================
if [[ $* == *--install-cuda* ]] ; then
  # bash中高亮显示
  highlight "Installing CUDA..."
  # Get ubuntu version:
  sudo apt-get install -y wget software-properties-common
  source /etc/lsb-release # fetch DISTRIB_CODENAME
  # ubuntu 14.04
  if [[ $DISTRIB_CODENAME == *"trusty"* ]] ; then
    # CUDA
    # wget下载
    wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.5-18_amd64.deb
    # dpkg安装
    sudo dpkg -i cuda-repo-ubuntu1404_7.5-18_amd64.deb
    rm cuda-repo-ubuntu1404_7.5-18_amd64.deb
    sudo apt-get update > /dev/null
    sudo apt-get install -y cuda-7-5
  # ubuntu 15.04
  elif [[ $DISTRIB_CODENAME == *"vivid"* ]] ; then
    # CUDA
    # wget下载
    wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1504/x86_64/cuda-repo-ubuntu1504_7.5-18_amd64.deb
    # dpkg安装
    sudo dpkg -i cuda-repo-ubuntu1504_7.5-18_amd64.deb
    rm cuda-repo-ubuntu1504_7.5-18_amd64.deb
    sudo apt-get update > /dev/null
    sudo apt-get install cuda-7-5
  # ubuntu 16.04
  elif [[ $DISTRIB_CODENAME == *"xenial"* ]]; then
    wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.44-1_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu1604_8.0.44-1_amd64.deb
    rm cuda-repo-ubuntu1604_8.0.44-1_amd64.deb
    sudo apt-get update > /dev/null
    sudo apt-get install -y cuda-8-0
  else
    echo "$DISTRIB_CODENAME is not yet supported"
    exit 1
  fi
fi # --install-cuda


# virtualenv 创建 虚拟环境 来安装项目依赖程序
# Create virtual python environment and install packages
highlight "Setting up virtual python environment..."
virtualenv python-environment
source python-environment/bin/activate
pip3 install pip --upgrade
# tensorflow 1.8.0 GPU版本
pip3 install tensorflow-gpu==1.8.0
# Python语言用于数字图像处理 
# scikit-image 是基于scipy的一款图像处理包，它将图片作为numpy数组进行处理，正好与matlab一样
pip3 install scikit-image
# Keras:基于Python的深度学习库
# Keras是一个高层神经网络API，Keras由纯Python编写而成并基Tensorflow、Theano以及CNTK后端
pip3 install keras
# IPython:一种交互式计算和开发环境的笔记
# 是一个 python 的交互式 shell，比默认的python shell 好用得多，支持变量自动补全，自动缩进，
# 支持 bash shell 命令，内置了许多很有用的功能和函数。
pip3 install IPython
# h5py文件是存放两类对象的容器，数据集(dataset)和组(group)，dataset类似数组类的数据集合
# HDF（Hierarchical Data Format）指一种为存储和处理大容量科学数据设计的文件格式及相应库文件。
pip3 install h5py
# Cython是让Python脚本支持C语言扩展的编译器
# Cython能够将Python+C混合编码的.pyx脚本转换为C代码
# https://github.com/Ewenwan/EasonCodeShare/tree/master/cython_tutorials/hello_world
pip3 install cython
# imgaug是一个封装好的用来进行图像augmentation的python库,支持关键点(keypoint)和bounding box一起变换。
# 数据增强 数据增多
pip3 install imgaug
# opencv-python接口
pip3 install opencv-python
# 符号链接 软件链接
ln -s python-environment/lib/python3.5/site-packages/numpy/core/include/numpy Core/Segmentation/MaskRCNN


# 创建项目
if [[ $* == *--build-dependencies* ]] ; then

  # Build dependencies
  mkdir -p deps
  cd deps
  
  # 编译opencv====================
  highlight "Building opencv..."
  git_clone "git clone --branch 3.4.1 --depth=1 https://github.com/opencv/opencv.git"
  cd opencv
  mkdir -p build
  cd build
  cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="`pwd`/../install" \
    \
    `# OpenCV: (building is not possible when DBUILD_opencv_video/_videoio is OFF?)` \
    -DWITH_CUDA=OFF  \
    -DBUILD_DOCS=OFF  \
    -DBUILD_PACKAGE=OFF \
    -DBUILD_TESTS=OFF  \
    -DBUILD_PERF_TESTS=OFF  \
    -DBUILD_opencv_apps=OFF \
    -DBUILD_opencv_calib3d=OFF  \
    -DBUILD_opencv_cudaoptflow=OFF  \
    -DBUILD_opencv_dnn=OFF  \
    -DBUILD_opencv_dnn_BUILD_TORCH_IMPORTER=OFF  \
    -DBUILD_opencv_features2d=OFF \
    -DBUILD_opencv_flann=OFF \
    -DBUILD_opencv_java=OFF  \
    -DBUILD_opencv_objdetect=OFF  \
    -DBUILD_opencv_python2=OFF  \
    -DBUILD_opencv_python3=OFF  \
    -DBUILD_opencv_photo=OFF \
    -DBUILD_opencv_stitching=OFF  \
    -DBUILD_opencv_superres=OFF  \
    -DBUILD_opencv_shape=OFF  \
    -DBUILD_opencv_videostab=OFF \
    -DBUILD_PROTOBUF=OFF \
    -DWITH_1394=OFF  \
    -DWITH_GSTREAMER=OFF  \
    -DWITH_GPHOTO2=OFF  \
    -DWITH_MATLAB=OFF  \
    -DWITH_NVCUVID=OFF \
    -DWITH_OPENCL=OFF \
    -DWITH_OPENCLAMDBLAS=OFF \
    -DWITH_OPENCLAMDFFT=OFF \
    -DWITH_TIFF=OFF  \
    -DWITH_VTK=OFF  \
    -DWITH_WEBP=OFF  \
    ..
  make -j8
  cd ../build
  OpenCV_DIR=$(pwd)
  cd ../..
  
# 编译 boost==========================================
  if [ -z "${BOOST_ROOT}" -a ! -d boost ]; then
    highlight "Building boost..."
    wget --no-clobber -O boost_1_62_0.tar.bz2 https://sourceforge.net/projects/boost/files/boost/1.62.0/boost_1_62_0.tar.bz2/download
    tar -xjf boost_1_62_0.tar.bz2 > /dev/null
    rm boost_1_62_0.tar.bz2
    cd boost_1_62_0
    mkdir -p ../boost
    ./bootstrap.sh --prefix=../boost
    ./b2 --prefix=../boost --with-filesystem install > /dev/null
    cd ..
    rm -r boost_1_62_0
    BOOST_ROOT=$(pwd)/boost
  fi

# 编译可视化 pangolin ====================================
  # build pangolin
  highlight "Building pangolin..."
  git_clone "git clone --depth=1 https://github.com/stevenlovegrove/Pangolin.git"
  cd Pangolin
  git pull
  mkdir -p build
  cd build
  cmake -DAVFORMAT_INCLUDE_DIR="" -DCPP11_NO_BOOST=ON ..
  make -j8
  Pangolin_DIR=$(pwd)
  cd ../..

# 编译 OpenNI2 3D传感器 开发接口====================================
  # build OpenNI2
  highlight "Building openni2..."
  git_clone "git clone --depth=1 https://github.com/occipital/OpenNI2.git"
  cd OpenNI2
  git pull
  make -j8
  cd ..

# 可视化 封装 openGL
  # build freetype-gl-cpp
  highlight "Building freetype-gl-cpp..."
  git_clone "git clone --depth=1 --recurse-submodules https://github.com/martinruenz/freetype-gl-cpp.git"
  cd freetype-gl-cpp
  mkdir -p build
  cd build
  cmake -DBUILD_EXAMPLES=OFF -DCMAKE_INSTALL_PREFIX="`pwd`/../install" -DCMAKE_BUILD_TYPE=Release ..
  make -j8
  make install
  cd ../..

# 条件随机场 语义分割 https://blog.csdn.net/u012759136/article/details/52434826
# 全连接条件随机场(DenseCRF)
  # build DenseCRF, see: http://graphics.stanford.edu/projects/drf/
  highlight "Building densecrf..."
  git_clone "git clone --depth=1 https://github.com/martinruenz/densecrf.git"
  cd densecrf
  git pull
  mkdir -p build
  cd build
  cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -fPIC" \
    ..
  make -j8
  cd ../..
#   实时超像素分割============================
  # build gSLICr, see: http://www.robots.ox.ac.uk/~victor/gslicr/
  highlight "Building gslicr..."
  git_clone "git clone --depth=1 https://github.com/carlren/gSLICr.git"
  cd gSLICr
  git pull
  mkdir -p build
  cd build
  cmake \
    -DOpenCV_DIR="${OpenCV_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDA_HOST_COMPILER=/usr/bin/gcc-4.9 \
    -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -D_FORCE_INLINES" \
    ..
  make -j8
  cd ../..

# MaskRCNN   目标语义分割 实例分割==============
  # Prepare MaskRCNN and data
  highlight "Building mask-rcnn with ms-coco..."
  git_clone "git clone --depth=1 https://github.com/matterport/Mask_RCNN.git"
  git_clone "git clone --depth=1 https://github.com/waleedka/coco.git"
  cd coco/PythonAPI
  make
  make install # Make sure to source the correct python environment first
  cd ../..
  cd Mask_RCNN
  mkdir -p data
  cd data
  wget --no-clobber https://github.com/matterport/Mask_RCNN/releases/download/v1.0/mask_rcnn_coco.h5
  cd ../..

  cd ..
fi # --build-dependencies

if [ -z "${BOOST_ROOT}" -a -d deps/boost ]; then
  BOOST_ROOT=$(pwd)/deps/boost
fi


# 编译最终的项目  动态slam  动态目标检测跟踪重建
# Build MaskFusion
highlight "Building MaskFusion..."
mkdir -p build
cd build
ln -s ../deps/Mask_RCNN ./ || true # Also, make sure that the file 'mask_rcnn_model.h5' is linked or present
cmake \
  -DBOOST_ROOT="${BOOST_ROOT}" \
  -DOpenCV_DIR="$(pwd)/../deps/opencv/build" \
  -DPangolin_DIR="$(pwd)/../deps/Pangolin/build/src" \
  -DMASKFUSION_PYTHON_VE_PATH="$(pwd)/../python-environment" \
  -DWITH_FREENECT2=OFF \
  ..
make -j8
cd ..
