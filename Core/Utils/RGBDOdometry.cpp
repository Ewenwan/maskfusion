/*
 * This file is part of ElasticFusion.
 *  ICP配准+rgb颜色匹配 跟踪
 *
 */

#include "RGBDOdometry.h"

RGBDOdometry::RGBDOdometry(int width, int height, 
                           float cx, float cy, 
                           float fx, float fy, 
                           unsigned char mask, 
                           float distThresh,
                           float angleThresh)
    : lastICPError(0),
      lastICPCount(width * height),
      lastRGBError(0),
      lastRGBCount(width * height),
      lastSO3Error(0),
      lastSO3Count(width * height),
      lastA(Eigen::Matrix<double, 6, 6, Eigen::RowMajor>::Zero()),
      lastb(Eigen::Matrix<double, 6, 1>::Zero()),
      sobelSize(3),
      sobelScale(1.0 / pow(2.0, sobelSize)),
      maxDepthDeltaRGB(0.07),
      maxDepthRGB(6.0),
      distThres_(distThresh),
      angleThres_(angleThresh),
      width(width),
      height(height),
      cx(cx),
      cy(cy),
      fx(fx),
      fy(fy),
      maskID(mask) 
{
    sumDataSE3.create(MAX_THREADS);
    outDataSE3.create(1);
    sumResidualRGB.create(MAX_THREADS);

    sumDataSO3.create(MAX_THREADS);
    outDataSO3.create(1);

    for (int i = 0; i < NUM_PYRS; i++) // 金字塔下采样 个层级尺寸
    {
        int2 nextDim = {height >> i, width >> i};
        pyrDims.push_back(nextDim);
    }

    for (int i = 0; i < NUM_PYRS; i++) 
    {
        lastDepth[i].create(pyrDims.at(i).x, pyrDims.at(i).y);// 深度图
        lastImage[i].create(pyrDims.at(i).x, pyrDims.at(i).y);// 灰度图像
        lastMask[i].create(pyrDims.at(i).x, pyrDims.at(i).y); // 掩码图

        nextDepth[i].create(pyrDims.at(i).x, pyrDims.at(i).y);
        nextImage[i].create(pyrDims.at(i).x, pyrDims.at(i).y);
        nextMask[i].create(pyrDims.at(i).x, pyrDims.at(i).y);

        lastNextImage[i].create(pyrDims.at(i).x, pyrDims.at(i).y);

        nextdIdx[i].create(pyrDims.at(i).x, pyrDims.at(i).y);// 灰度水平梯度
        nextdIdy[i].create(pyrDims.at(i).x, pyrDims.at(i).y);// 灰度垂直梯度

        pointClouds[i].create(pyrDims.at(i).x, pyrDims.at(i).y);// 3D点云图

        corresImg[i].create(pyrDims.at(i).x, pyrDims.at(i).y);
    }

    intr.cx = cx;// 相机内参数
    intr.cy = cy;
    intr.fx = fx;
    intr.fy = fy;

    iterations.reserve(NUM_PYRS);

    vmaps_g_prev_.resize(NUM_PYRS);// 2d 地图       xyz间隔行存放
    nmaps_g_prev_.resize(NUM_PYRS);// 2d 归一化地图

    //  vmaps_curr_.resize(NUM_PYRS);
    //  nmaps_curr_.resize(NUM_PYRS);

    for (int i = 0; i < NUM_PYRS; ++i) 
    {
        int pyr_rows = height >> i;
        int pyr_cols = width >> i;

        vmaps_g_prev_[i].create(pyr_rows * 3, pyr_cols);//  2d 地图 xyz间隔行存放 行数扩大3倍
        nmaps_g_prev_[i].create(pyr_rows * 3, pyr_cols);

        //    vmaps_curr_[i].create(pyr_rows * 3, pyr_cols);
        //    nmaps_curr_[i].create(pyr_rows * 3, pyr_cols);
    }

    vmaps_tmp.create(height * 4 * width);
    nmaps_tmp.create(height * 4 * width);

    minimumGradientMagnitudes.reserve(NUM_PYRS);
    minimumGradientMagnitudes[0] = 5;
    minimumGradientMagnitudes[1] = 3;
    minimumGradientMagnitudes[2] = 1;
}

RGBDOdometry::~RGBDOdometry() {}

//void RGBDOdometry::initICP(const std::vector<DeviceArray2D<float> >& depthPyramid,
//                           const std::vector<DeviceArray2D<unsigned char> >& maskPyramid, const float depthCutoff) {
//  for (int i = 0; i < RGBDOdometry::NUM_PYRS; ++i) {
//    createVMap(intr(i), depthPyramid[i], vmaps_curr_[i], depthCutoff);
//    createNMap(vmaps_curr_[i], nmaps_curr_[i]);
//  }

//  cudaDeviceSynchronize();
//}

void RGBDOdometry::initICP(const std::vector<DeviceArray2D<float>>* vertexMapPyramid,
                           const std::vector<DeviceArray2D<float>>* normalMapPyramid,
                           const std::vector<DeviceArray2D<unsigned char>>* prevMaskPyramid)
{
    this->vertexMapPyramid = vertexMapPyramid;
    this->normalMapPyramid = normalMapPyramid;
    this->prevMaskPyramid = prevMaskPyramid;
}

//void RGBDOdometry::generateCurrentMaps(GPUTexture* predictedVertices, GPUTexture* predictedNormals, const float depthCutoff) {
//  cudaArray* textPtr;

//  predictedVertices->cudaMap();
//  textPtr = predictedVertices->getCudaArray();
//  cudaMemcpyFromArray(vmaps_tmp.ptr(), textPtr, 0, 0, vmaps_tmp.sizeBytes(), cudaMemcpyDeviceToDevice);
//  predictedVertices->cudaUnmap();

//  predictedNormals->cudaMap();
//  textPtr = predictedNormals->getCudaArray();
//  cudaMemcpyFromArray(nmaps_tmp.ptr(), textPtr, 0, 0, nmaps_tmp.sizeBytes(), cudaMemcpyDeviceToDevice);
//  predictedNormals->cudaUnmap();

//  copyMaps(vmaps_tmp, nmaps_tmp, vmaps_curr_[0], nmaps_curr_[0]);

//  for (int i = 1; i < NUM_PYRS; ++i) {
//    resizeVMap(vmaps_curr_[i - 1], vmaps_curr_[i]);
//    resizeNMap(nmaps_curr_[i - 1], nmaps_curr_[i]);
//  }

//  cudaDeviceSynchronize();
//}



void RGBDOdometry::initICPModel(GPUTexture* predictedVertices, 
                                GPUTexture* predictedNormals, 
                                const float depthCutoff,
                                const Eigen::Matrix4f& modelPose)
{
    cudaArray* textPtr;

    predictedVertices->cudaMap();
    textPtr = predictedVertices->getCudaArray();
    cudaMemcpyFromArray(vmaps_tmp.ptr(), 
                        textPtr, 0, 0, vmaps_tmp.sizeBytes(), cudaMemcpyDeviceToDevice);
    predictedVertices->cudaUnmap();

    predictedNormals->cudaMap();
    textPtr = predictedNormals->getCudaArray();
    cudaMemcpyFromArray(nmaps_tmp.ptr(), textPtr, 0, 0, nmaps_tmp.sizeBytes(), cudaMemcpyDeviceToDevice);
    predictedNormals->cudaUnmap();

    copyMaps(vmaps_tmp, nmaps_tmp, vmaps_g_prev_[0], nmaps_g_prev_[0]);

    for (int i = 1; i < NUM_PYRS; ++i)
    {
        resizeVMap(vmaps_g_prev_[i - 1], vmaps_g_prev_[i]);
        resizeNMap(nmaps_g_prev_[i - 1], nmaps_g_prev_[i]);
    }

    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rcam = modelPose.topLeftCorner(3, 3);
    Eigen::Vector3f tcam = modelPose.topRightCorner(3, 1);

    mat33 device_Rcam = Rcam;
    float3 device_tcam = *reinterpret_cast<float3*>(tcam.data());

    for (int i = 0; i < NUM_PYRS; ++i) {
        tranformMaps(vmaps_g_prev_[i], 
                     nmaps_g_prev_[i], 
                     device_Rcam, 
                     device_tcam, 
                     vmaps_g_prev_[i],
                     nmaps_g_prev_[i]);
    }

    cudaDeviceSynchronize();
}

void RGBDOdometry::populateRGBDData(GPUTexture* rgb, 
                                    DeviceArray2D<float>* destDepths, 
                                    DeviceArray2D<unsigned char>* destImages,
                                    DeviceArray2D<unsigned char>* destMasks) 
{
    verticesToDepth(vmaps_tmp, destDepths[0], maxDepthRGB);
    
  // 深度图 5×5高斯下采样=====================
    for (int i = 0; i + 1 < NUM_PYRS; i++) pyrDownGaussF(destDepths[i], destDepths[i + 1]);

    rgb->cudaMap();
    cudaArray* textPtr = rgb->getCudaArray();
    imageBGRToIntensity(textPtr, destImages[0]);
    rgb->cudaUnmap();

    for (int i = 0; i + 1 < NUM_PYRS; i++) 
    {
        pyrDownUcharGauss(destImages[i], destImages[i + 1]);
        pyrDownUcharGauss(destMasks[i], destMasks[i + 1]);
    }

    cudaDeviceSynchronize();
}

void RGBDOdometry::initRGBModel(GPUTexture* rgb) {
    // NOTE: This depends on vmaps_tmp containing the corresponding depth from initICPModel
    populateRGBDData(rgb, &lastDepth[0], &lastImage[0], &lastMask[0]);
}

void RGBDOdometry::initRGB(GPUTexture* rgb) {
    // NOTE: This depends on vmaps_tmp containing the corresponding depth from initICP
    populateRGBDData(rgb, &nextDepth[0], &nextImage[0], &nextMask[0]);
}

void RGBDOdometry::initFirstRGB(GPUTexture* rgb) {
    rgb->cudaMap();
    cudaArray* textPtr = rgb->getCudaArray();
    imageBGRToIntensity(textPtr, lastNextImage[0]);
    rgb->cudaUnmap();

    for (int i = 0; i + 1 < NUM_PYRS; i++) {
        pyrDownUcharGauss(lastNextImage[i], lastNextImage[i + 1]);
    }
}

Eigen::Matrix4f RGBDOdometry::getIncrementalTransformation(Eigen::Vector3f& trans,
                                                           Eigen::Matrix<float, 3, 3, Eigen::RowMajor>& rot,
                                                           const bool& rgbOnly,
                                                           const float& icpWeight,
                                                           const bool& pyramid,
                                                           const bool& fastOdom,
                                                           const bool& so3,
                                                           const cudaSurfaceObject_t& icpErrorSurface,
                                                           const cudaSurfaceObject_t& rgbErrorSurface)
{
    bool icp = !rgbOnly && icpWeight > 0;
    bool rgb = rgbOnly || icpWeight < 100;

    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rprev = rot;
    Eigen::Vector3f tprev = trans;

    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rcurr = Rprev;
    Eigen::Vector3f tcurr = tprev;

    if (rgb) {
        for (int i = 0; i < NUM_PYRS; i++) {
            // sobelGaussian(nextImage[i], nextdIdx[i], nextdIdy[i]);
          // 计算灰度图像 的 水平和 垂直梯度====================================
            computeDerivativeImages(nextImage[i], nextdIdx[i], nextdIdy[i]);
        }
    }

    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> resultR = 
        Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Identity();

    if (so3) {
        int pyramidLevel = 2;

        Eigen::Matrix<float, 3, 3, Eigen::RowMajor> R_lr = 
           Eigen::Matrix<float, 3, 3, Eigen::RowMajor>::Identity();

      // 相机内参数
        Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K = 
           Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Zero();
      
        K(0, 0) = intr(pyramidLevel).fx;
        K(1, 1) = intr(pyramidLevel).fy;
        K(0, 2) = intr(pyramidLevel).cx;
        K(1, 2) = intr(pyramidLevel).cy;
        K(2, 2) = 1;

        float lastError = std::numeric_limits<float>::max() / 2;
        float lastCount = std::numeric_limits<float>::max() / 2;

        Eigen::Matrix<double, 3, 3, Eigen::RowMajor> lastResultR = 
            Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Identity();

        for (int i = 0; i < 10; i++) 
        {
            Eigen::Matrix<float, 3, 3, Eigen::RowMajor> jtj;
            Eigen::Matrix<float, 3, 1> jtr;

            Eigen::Matrix<double, 3, 3, Eigen::RowMajor> homography = K * resultR * K.inverse();

            mat33 imageBasis;
            memcpy(&imageBasis.data[0], homography.cast<float>().eval().data(), sizeof(mat33));

            Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K_inv = K.inverse();
            mat33 kinv;
            memcpy(&kinv.data[0], K_inv.cast<float>().eval().data(), sizeof(mat33));

            Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K_R_lr = K * resultR;
            mat33 krlr;
            memcpy(&krlr.data[0], K_R_lr.cast<float>().eval().data(), sizeof(mat33));

            float residual[2];

            TICK("so3Step");
            so3Step(lastNextImage[pyramidLevel], 
                    nextImage[pyramidLevel], 
                    imageBasis, kinv, krlr, 
                    sumDataSO3, 
                    outDataSO3, 
                    jtj.data(),
                    jtr.data(),
                    &residual[0], 
                    GPUConfig::getInstance().so3StepThreads, 
                    GPUConfig::getInstance().so3StepBlocks);
          
            TOCK("so3Step");

            lastSO3Error = sqrt(residual[0]) / residual[1];
            lastSO3Count = residual[1];

            // Converged
            if (lastSO3Error < lastError && fabs(lastError - lastSO3Count) < 0.001) 
            {
                break;
            } else if (lastSO3Error > lastError + 0.001) {  // Diverging
                lastSO3Error = lastError;
                lastSO3Count = lastCount;
                resultR = lastResultR;
                break;
            }

            lastError = lastSO3Error;
            lastCount = lastSO3Count;
            lastResultR = resultR;

            Eigen::Vector3f delta = jtj.ldlt().solve(jtr);

            Eigen::Matrix<double, 3, 3, Eigen::RowMajor> rotUpdate = OdometryProvider::rodrigues(delta.cast<double>());

            R_lr = rotUpdate.cast<float>() * R_lr;

            for (int x = 0; x < 3; x++) {
                for (int y = 0; y < 3; y++) {
                    resultR(x, y) = R_lr(x, y);
                }
            }
        }
    }

    iterations[0] = fastOdom ? 3 : 10;
    iterations[1] = pyramid ? 5 : 0;
    iterations[2] = pyramid ? 4 : 0;

    Eigen::Isometry3f transform;
    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rprev_inv = Rprev.inverse();
    mat33 device_Rprev_inv = Rprev_inv;
    float3 device_tprev = *reinterpret_cast<float3*>(tprev.data());

    Eigen::Matrix<double, 4, 4, Eigen::RowMajor> resultRt = Eigen::Matrix<double, 4, 4, Eigen::RowMajor>::Identity();

    if (so3) {
        for (int x = 0; x < 3; x++) {
            for (int y = 0; y < 3; y++) {
                resultRt(x, y) = resultR(x, y);
            }
        }
    }

    // Per pyramid level
    for (int i = NUM_PYRS - 1; i >= 0; i--) {
        if (rgb) {
            projectToPointCloud(lastDepth[i], pointClouds[i], intr, i);
        }

        Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Zero();

        K(0, 0) = intr(i).fx;
        K(1, 1) = intr(i).fy;
        K(0, 2) = intr(i).cx;
        K(1, 2) = intr(i).cy;
        K(2, 2) = 1;

        lastRGBError = std::numeric_limits<float>::max();

        // Optimization iterations
        for (int j = 0; j < iterations[i]; j++) 
        {
            Eigen::Matrix<double, 4, 4, Eigen::RowMajor> Rt = resultRt.inverse();

            Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R = Rt.topLeftCorner(3, 3);

            Eigen::Matrix<double, 3, 3, Eigen::RowMajor> KRK_inv = K * R * K.inverse();
            mat33 krkInv;
            memcpy(&krkInv.data[0], KRK_inv.cast<float>().eval().data(), sizeof(mat33));

            Eigen::Vector3d Kt = Rt.topRightCorner(3, 1);
            Kt = K * Kt;
            float3 kt = {(float)Kt(0), (float)Kt(1), (float)Kt(2)};

            int sigma = 0;
            int rgbSize = 0;

            if (rgb) {
                TICK("computeRgbResidual");
                computeRgbResidual(pow(minimumGradientMagnitudes[i], 2.0) / pow(sobelScale, 2.0), nextdIdx[i], nextdIdy[i], lastDepth[i],
                                   nextDepth[i], lastImage[i], nextImage[i], lastMask[i], nextMask[i], corresImg[i], sumResidualRGB,
                                   maxDepthDeltaRGB, kt, krkInv, sigma, rgbSize, GPUConfig::getInstance().rgbResThreads,
                                   GPUConfig::getInstance().rgbResBlocks,
                                   /*(i == 0 && j == iterations[i]-1) ? rgbErrorSurface :*/ 0, maskID);
                TOCK("computeRgbResidual");
            }

            float tmpError = sqrt(sigma) / rgbSize;
            float sigmaVal = (tmpError == 0) ? 1 : rgbSize;

            if (rgbOnly && tmpError > lastRGBError) {
                break;
            }

            lastRGBError = tmpError;
            lastRGBCount = rgbSize;

            if (rgbOnly) {
                sigmaVal = -1;  // Signals the internal optimisation to weight evenly
            }

            Eigen::Matrix<float, 6, 6, Eigen::RowMajor> A_icp;
            Eigen::Matrix<float, 6, 1> b_icp;

            mat33 device_Rcurr = Rcurr;
            float3 device_tcurr = *reinterpret_cast<float3*>(tcurr.data());

            // current frame data
            const DeviceArray2D<float>& vmap_curr = (*vertexMapPyramid)[i];//vmaps_curr_[i];
            const DeviceArray2D<float>& nmap_curr = (*normalMapPyramid)[i];//nmaps_curr_[i];

            // model data
            DeviceArray2D<float>& vmap_g_prev = vmaps_g_prev_[i];
            DeviceArray2D<float>& nmap_g_prev = nmaps_g_prev_[i];

            float residual[2];

            if (icp) {
                TICK("icpStep");
                icpStep(device_Rcurr, device_tcurr, vmap_curr, nmap_curr, device_Rprev_inv, device_tprev, intr(i), vmap_g_prev, nmap_g_prev,
                        distThres_, angleThres_, sumDataSE3, outDataSE3, A_icp.data(), b_icp.data(), &residual[0],
                        GPUConfig::getInstance().icpStepThreads, GPUConfig::getInstance().icpStepBlocks,
                        (i == 0 && j == iterations[i] - 1) ? icpErrorSurface : 0, (*prevMaskPyramid)[i], 0);
                TOCK("icpStep");
            }

            lastICPError = sqrt(residual[0]) / residual[1];
            lastICPCount = residual[1];

            Eigen::Matrix<float, 6, 6, Eigen::RowMajor> A_rgbd;
            Eigen::Matrix<float, 6, 1> b_rgbd;

            if (rgb) {
                TICK("rgbStep");
                rgbStep(corresImg[i], sigmaVal, pointClouds[i], intr(i).fx, intr(i).fy, nextdIdx[i], nextdIdy[i], sobelScale, sumDataSE3,
                        outDataSE3, A_rgbd.data(), b_rgbd.data(), GPUConfig::getInstance().rgbStepThreads, GPUConfig::getInstance().rgbStepBlocks);
                TOCK("rgbStep");
            }

            Eigen::Matrix<double, 6, 1> result;
            Eigen::Matrix<double, 6, 6, Eigen::RowMajor> dA_rgbd = A_rgbd.cast<double>();
            Eigen::Matrix<double, 6, 6, Eigen::RowMajor> dA_icp = A_icp.cast<double>();
            Eigen::Matrix<double, 6, 1> db_rgbd = b_rgbd.cast<double>();
            Eigen::Matrix<double, 6, 1> db_icp = b_icp.cast<double>();

            if (icp && rgb) {
                double w = icpWeight;
                lastA = dA_rgbd + w * w * dA_icp;
                lastb = db_rgbd + w * db_icp;
                result = lastA.ldlt().solve(lastb);
            } else if (icp) {
                lastA = dA_icp;
                lastb = db_icp;
                result = lastA.ldlt().solve(lastb);
            } else if (rgb) {
                lastA = dA_rgbd;
                lastb = db_rgbd;
                result = lastA.ldlt().solve(lastb);
            } else {
                assert(false && "Control shouldn't reach here");
            }

            Eigen::Isometry3f currentT;
            OdometryProvider::computeUpdateSE3(resultRt, result, transform);

            currentT.setIdentity();
            currentT.rotate(Rprev);
            currentT.translation() = tprev;

            currentT = currentT * transform.inverse();

            tcurr = currentT.translation();
            Rcurr = currentT.rotation();
        }
    }

    if (rgb && (tcurr - tprev).norm() > 0.3) {
        Rcurr = Rprev;
        tcurr = tprev;
        transform.setIdentity();
    }

    if (so3) {
        for (int i = 0; i < NUM_PYRS; i++) {
            std::swap(lastNextImage[i], nextImage[i]);
        }
    }

    trans = tcurr;
    rot = Rcurr;

    Eigen::Matrix4f retVal = Eigen::Matrix4f::Identity();
    retVal.topRightCorner(3, 1) = transform.translation();
    retVal.topLeftCorner(3, 3) = transform.rotation();
    return retVal;
}

Eigen::MatrixXd RGBDOdometry::getCovariance() { return lastA.cast<double>().lu().inverse(); }
