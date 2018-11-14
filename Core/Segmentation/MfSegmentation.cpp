/*
 * This file is part of https://github.com/martinruenz/maskfusion
 *  MaskFusion 方法分割
  1. 距离、凸凹性edgeMap浮点边缘图 
  2. 阈值二值化  255/0
  3. 腐蚀膨胀
  4. 反向       255-x
  5. 根据目标检测语义分割，剔除部分 不考虑的物体，例如人
  6. opencv 联通域分析
 */

#include "MfSegmentation.h"
#include "GPUTexture.h"
#include "Model/Model.h"
#include "Model/GlobalProjection.h"
#include "Cuda/segmentation.cuh"
#include "MaskRCNN/MaskRCNN.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <memory>
#include <algorithm>

//#define SHOW_DEBUG_VISUALISATION
//#define WRITE_MASK_FILES
#ifdef WRITE_MASK_FILES
    int WRITE_MASK_INDEX = 0;
    const std::string WRITE_MASK_DIR = "/tmp/mf";
#endif

MfSegmentation::MfSegmentation(int w, int h,
                                       const CameraModel& cameraIntrinsics,
                                       bool embedMaskRCNN,
                                       std::shared_ptr<GPUTexture> textureRGB,
                                       std::shared_ptr<GPUTexture> textureDepthMetric,
                                       GlobalProjection* globalProjection,
                                       std::queue<FrameDataPointer>* queue) :
    minMaskModelOverlap(0.05f), 
    minMappedComponentSize(160), 
    minNewMaskPixels(7000/*2000*/), 
    REUSE_FILTERED_MAPS(true) 
{

    floatEdgeMap.create(h, w);// 边缘图
    floatBuffer.create(h, w);

    ucharBuffer.create(h, w);
    binaryEdgeMap.create(h, w);

    cv8UC1Buffer.create(h, w, CV_8UC1);
    cvLabelComps.create(h, w, CV_32S);
    cvLabelEdges.create(h, w, CV_32S);
    semanticIgnoreMap = cv::Mat::zeros(h, w, CV_8UC1);

    if(!REUSE_FILTERED_MAPS)
    {
        this->textureDepthMetric = textureDepthMetric;
        this->textureRGB = textureRGB;
        this->cameraIntrinsics = cameraIntrinsics;
        vertexMap.create(h*3,w);
        normalMap.create(h*3,w);
        depthMapMetric.create(h,w);
        depthMapMetricFiltered.create(h,w);
        rgb.create(h,w);
    }

    segmentationMap = std::make_shared<GPUTexture>(w, h, GL_R32F, 
                                                   GL_RED, GL_FLOAT, 
                                                   false, true, 
                                                   cudaGraphicsRegisterFlagsSurfaceLoadStore);
    //debugMap = std::make_shared<GPUTexture>(w, h, GL_R32F, GL_RED, GL_FLOAT, true, true, cudaGraphicsRegisterFlagsSurfaceLoadStore);
    allocateModelBuffers(5);
    maskToID[255] = 255; // Ignored
    maskToID[0] = 0; // Background

    if(embedMaskRCNN)
    {
        maskRCNN = std::make_unique<MaskRCNN>(queue);
        sequentialMaskRCNN = (queue == nullptr);
    }

    this->globalProjection = globalProjection;
}

MfSegmentation::~MfSegmentation(){}

// models 哪里过来的????????
SegmentationResult MfSegmentation::performSegmentation(std::list<std::shared_ptr<Model> > &models,
                                                       FrameDataPointer frame,
                                                       unsigned char nextModelID,
                                                       bool allowNew)
{
    TICK("segmentation");
#ifdef SHOW_DEBUG_VISUALISATION
    // 颜色
    const unsigned char colors[31][3] = {
        {0, 0, 0},     {0, 0, 255},     {255, 0, 0},   {0, 255, 0},     
        {255, 26, 184},  {255, 211, 0},   {0, 131, 246},  {0, 140, 70},
        {167, 96, 61}, {79, 0, 105},    {0, 255, 246}, {61, 123, 140},  
        {237, 167, 255}, {211, 255, 149}, {184, 79, 255}, {228, 26, 87},
        {131, 131, 0}, {0, 255, 149},   {96, 0, 43},   {246, 131, 17},  
        {202, 255, 0},   {43, 61, 0},     {0, 52, 193},   {255, 202, 131},
        {0, 43, 96},   {158, 114, 140}, {79, 184, 17}, {158, 193, 255}, 
        {149, 158, 123}, {255, 123, 175}, {158, 8, 0}};
    auto getColor = [&colors](unsigned index) -> cv::Vec3b 
    {
        return (index == 255) ? cv::Vec3b(255, 255, 255) : (cv::Vec3b)colors[index % 31];
    };

    auto mapLabelToColorImage = [&getColor](cv::Mat input, bool white0 = false) -> cv::Mat
    {
        std::function<cv::Vec3b(unsigned)> getIndex;
        auto getColorWW = [&](unsigned index) -> cv::Vec3b 
        { 
            return (white0 && index == 0) ? cv::Vec3b(255, 255, 255) : getColor(index); 
        };
        if (input.type() == CV_32SC1)
            getIndex = [&](unsigned i) -> cv::Vec3b { return getColorWW(input.at<int>(i)); };
        
        else if (input.type() == CV_8UC1)
            getIndex = [&](unsigned i) -> cv::Vec3b { return getColorWW(input.data[i]); };
        else
            assert(0);
        cv::Mat result(input.rows, input.cols, CV_8UC3);
        for (unsigned i = 0; i < result.total(); ++i) 
        {
            ((cv::Vec3b*)result.data)[i] = getIndex(i);
        }
        return result;
    };

    auto overlayMask = [&getColor](cv::Mat rgb, cv::Mat mask) -> cv::Mat
    {
        cv::Mat vis(rgb.rows, rgb.cols, CV_8UC3);
        for (unsigned i = 0; i < rgb.total(); ++i) 
        {
            vis.at<cv::Vec3b>(i) = getColor(mask.data[i]);
            vis.at<cv::Vec3b>(i) = 0.5 * vis.at<cv::Vec3b>(i) + 0.5 * rgb.at<cv::Vec3b>(i);
        }
        return vis;
    };
  
    if(frame->mask.total()) 
    {
        cv::imshow("maskrcnn", overlayMask(frame->rgb, frame->mask));// RGB图像上  加上 0.5 可见度 的 mask
        cv::waitKey(1);
    }
#endif

    if(frame->mask.total() == 0)// 未检测到物体================================
    {
        if(!maskRCNN) throw std::runtime_error("MaskRCNN is not embedded and no masks were pre-computed.");
        else if(sequentialMaskRCNN) maskRCNN->executeSequential(frame);// 使用maskRCNN检测物体
    }

#ifdef WRITE_MASK_FILES
    cv::imwrite(WRITE_MASK_DIR + "mrcnn" + std::to_string(frame->index) + ".png", frame->mask);
#endif

    SegmentationResult result;
    const int& width = frame->depth.cols;
    const int& height = frame->depth.rows;
    const size_t total = frame->depth.total();
    result.fullSegmentation = cv::Mat::zeros(height, width, CV_8UC1);
    const int nMasks = int(frame->classIDs.size());
    const int nModels = int(models.size());

    // Prepare data (vertex/depth/... maps)
    TICK("segmentation-geom");
    if(REUSE_FILTERED_MAPS)
    {
        Model::GPUSetup& gpu = Model::GPUSetup::getInstance();
        
        //利用 周围9点的 距离、凸凹性计算点的边缘属性 进而 进行分割==========================
        computeGeometricSegmentationMap(gpu.vertex_map_tmp[0], 
                                        gpu.normal_map_tmp[0], 
                                        floatEdgeMap, 
                                        weightDistance,
                                        weightConvexity);
    } 
    else
    {
        computeLookups();
        computeGeometricSegmentationMap(vertexMap, 
                                        normalMap,
                                        floatEdgeMap, 
                                        weightDistance, 
                                        weightConvexity);
    }
    TOCK("segmentation-geom");

    // Prepare per model data (ICP texture, conf texture...)
    allocateModelBuffers(nModels+1);
    auto modelItr = models.begin();
    cv::Mat projectedIDs = globalProjection->getProjectedModelIDs();

#ifdef SHOW_DEBUG_VISUALISATION
    static int DV_CNT = 0;
    cv::imshow( "Projected IDs", mapLabelToColorImage(projectedIDs) );
#endif
#ifdef WRITE_MASK_FILES
    cv::imwrite(WRITE_MASK_DIR + "projected-label" + std::to_string(frame->index) + ".png", projectedIDs);
#endif

    // TODO: Also fix "downloadDirect"
    //cv::Mat projectedDepth = globalProjection->getProjectedDepth(); // TODO remove and perform relevant steps directly on the GPU, this can save time!

    TICK("segmentation-DL");
    for (unsigned char m = 0; m < models.size(); ++m,++modelItr) 
    {
        ModelBuffers& mBuffers = modelBuffers[m];
        auto& model = *modelItr;

        mBuffers.modelID = model->getID();

        SegmentationResult::ModelData modelData(model->getID());
        modelData.modelListIterator = modelItr;
        modelData.depthMean = 30; // FIXME, this requirement is not intuitive!!
        modelData.depthStd = 30;
        result.modelData.push_back(modelData);
        modelIDToIndex[model->getID()] = m;
    }
    if (allowNew) 
    {
        modelIDToIndex[nextModelID] = models.size();
        modelBuffers[models.size()].modelID = nextModelID;
    }
    TOCK("segmentation-DL");

  
  
    // Perform geometric segmentation

    TICK("segmentation-geom-post");
    DeviceArray2D<float>& edgeMap = floatEdgeMap;

    // Copy edge-map to segmentationMap for visualisation
    cudaArray* segmentationMapPtr = segmentationMap->getCudaArray();
    cudaMemcpy2DToArray(segmentationMapPtr, 0, 0, 
                        edgeMap.ptr(), edgeMap.step(), 
                        edgeMap.colsBytes(), edgeMap.rows(),
                        cudaMemcpyDeviceToDevice);
    // edgeMap浮点边缘图 ---> 阈值二值化  threshold --> 二值边缘图 ----> binaryEdgeMap
    thresholdMap(edgeMap, binaryEdgeMap, threshold);// 255/0
    
    // 使用 1次膨胀、腐蚀 来对二值边缘图 进行 分割================
    morphGeometricSegmentationMap(binaryEdgeMap,ucharBuffer, 
                                  morphEdgeRadius, morphEdgeIterations);
    // 反向======================================
    invertMap(binaryEdgeMap,ucharBuffer);// 255-x
    ucharBuffer.download(cv8UC1Buffer.data, ucharBuffer.cols()); // 分割最终结果为 cv8UC1Buffer

#ifdef SHOW_DEBUG_VISUALISATION
    cv::Mat vis(480,640,CV_8UC3);
    for (int i = 0; i < 640*480; ++i) // 图像大小固定！！！！！！！！！！！！！！！！！！！
    {
        float f = cv8UC1Buffer.data[i] / 255.0f;// 0/1  0为不需要的
        // 要么是 (255,255,255)  要么是 (0,0,255) ==========================================
        vis.at<cv::Vec3b>(i) = f * cv::Vec3b(255,255,255) + (1-f) * cv::Vec3b(0,0,255);
    }
    cv::imshow("Geometric edges", vis);// vis 几何边缘===============================
    cv::waitKey(1);
#endif

    // Build use ignore map
    if(nMasks)
    {
        for(size_t i=0; i<total; i++)
        {
// mask 存储的为像素点 所属的类别ID ==========================================
            
            if(frame->classIDs[frame->mask.data[i]] == personClassID) // 过滤掉 mask为人的区域!!!!!!!!!!!!!
            {
                semanticIgnoreMap.data[i] = 255;// 忽略
                cv8UC1Buffer.data[i] = 0;// 0为不需要的
            } 
            else 
            {
                semanticIgnoreMap.data[i] = 0;
            }
        }
        //cv::compare(frame->mask, cv::Scalar(...), semanticIgnoreMap, CV_CMP_EQ);
    }
    else 
    {
        for(size_t i=0; i<total; i++)
        {
            if(semanticIgnoreMap.data[i]) cv8UC1Buffer.data[i] = 0;// semanticIgnoreMap255为忽略  cv8UC1Buffer0为忽略
        }
    }

    // Run connected-components on segmented map
    cv::Mat statsComp, centroidsComp;
// 连通域处理函数
// nComponents 原始联通域总数
// 二值图像cv8UC1Buffer,   cvLabelComps二值图像标签, 
// 和原图一样大的标记图 stats, nComponents×5的矩阵 表示每个连通区域的 外接矩形(x,y,w,h) 和 面积（pixel） 
// centroidsComp ,          nComponents×2的矩阵 表示每个连通区域的质心
// https://blog.csdn.net/i_chaoren/article/details/78358297
    int nComponents = cv::connectedComponentsWithStats(cv8UC1Buffer, cvLabelComps, statsComp, centroidsComp, 4);
    TOCK("segmentation-geom-post");

    // Todo, this can be faster! (GPU?)
    if(removeEdges)
    {
        const bool remove_small_components = true;
        const int small_components_threshold = 50;
        const int removeEdgeIterations = 5;
        TICK("segmentation-removeedge");
#ifdef SHOW_DEBUG_VISUALISATION
        cv::imshow("Connected Components (before edge-removal)", mapLabelToColorImage(cvLabelComps) );
        cv::Mat re_vis; // 剩余的================
        frame->rgb.copyTo(re_vis);// 彩色图像
#endif
        auto checkNeighbor = [&, this](int y, int x, int& n, float d)
        {
            n = this->cvLabelComps.at<int>(y,x);
            // small_components_threshold 小目标区域 阈值 ===========================
            // 去除面积小于 small_components_threshold的连通域
            // 深度值差值较小
            if(n != 0 && std::fabs(frame->depth.at<float>(y,x)-d) < 0.008 && statsComp.at<int>(n, 4) > small_components_threshold)
            {
#ifdef SHOW_DEBUG_VISUALISATION
            re_vis.at<cv::Vec3b>(y,x) = cv::Vec3b(255,0,255);// BGR 粉红色?????
#endif
                return true;
            }
            return false;
        };
      
        for (int i = 0; i < removeEdgeIterations; ++i) 
        {
            cv::Mat r;
            cvLabelComps.copyTo(r);// cvLabelComps二值图像标签
            for (int y = 1; y < height-1; ++y) 
            { // TODO reduce index computations here
                for (int x = 1; x < width-1; ++x)
                {
                    int& c = r.at<int>(y,x);// 联通域 标签值==================
//                    statsComp.at<int>(c, 4);
                    float d = frame->depth.at<float>(y,x);// 对应深度值=======

                    if(c==0 || (remove_small_components && statsComp.at<int>(c, 4) < small_components_threshold))
                    {// 背景 或者 联通域面积太小===============================
                        int c2;
                      // 检查周围八点 是否是邻居点=======c2为0?????=======
                        if(checkNeighbor(y-1,x-1,c2,d)) { c = c2; continue; }
                        if(checkNeighbor(y-1,x,c2,d)) { c = c2; continue; }
                        if(checkNeighbor(y-1,x+1,c2,d)) { c = c2; continue; }
                        if(checkNeighbor(y,x-1,c2,d)) { c = c2; continue; }
                        if(checkNeighbor(y,x+1,c2,d)) { c = c2; continue; }
                        if(checkNeighbor(y+1,x-1,c2,d)) { c = c2; continue; }
                        if(checkNeighbor(y+1,x,c2,d)) { c = c2; continue; }
                        if(checkNeighbor(y+1,x+1,c2,d)) { c = c2; continue; }
                    }
                }
            }
            cvLabelComps = r;
        }
        TOCK("segmentation-removeedge");
#ifdef SHOW_DEBUG_VISUALISATION
        imshow("removed edges", re_vis);
#endif
    }
#ifdef SHOW_DEBUG_VISUALISATION
    cv::imshow( "Connected Components", mapLabelToColorImage(cvLabelComps) );
//    cv::imwrite(std::string("/tmp/outmf/cc") + std::to_string(DV_CNT) + ".png", mapLabelToColorImage(cvLabelComps)); std::cout << "!WRITING!" << std::endl;
#endif

// 目标检测mask 和 点云分割联通域图 进行关联===========================================
    // Assign mask to each component
    TICK("segmentation-assign");
  // 每个联通域 关联 目标检测mask========
    std::vector<int> mapComponentToMask(nComponents, 0); // By default, components are mapped to background (maskid==0)
  // 每个目标检测mask 关联 联通域大小的总面积========  一个mask可能关联多个联通域!!!!!!!!!!!!!
    std::vector<int> maskComponentPixels(nMasks, 0); // Number of pixels per mask
    std::vector<BoundingBox> maskComponentBoxes(nMasks);// mask 和 多个 联通域关联，所有联通域的矩形框 融合merge
                                                        // 一个mask可能关联多个联通域
  // 联通域 和 目标检测mask 重叠度 矩阵===
    cv::Mat compMaskOverlap(nComponents,nMasks,CV_32SC1, cv::Scalar(0));
//   cv::Mat compModelOverlap(nComponents,nModels,CV_32SC1, cv::Scalar(0));
  // nModels 是外面传递来的
    Eigen::MatrixXi compModelOverlap = Eigen::MatrixXi::Zero(nComponents, nModels);

// Compute component-model overlap
    for (size_t i = 0; i < total; ++i)// total = 480×640
        compModelOverlap(cvLabelComps.at<int>(i), projectedIDs.data[i])++;// 联通域id：模型id 数量+1
//        compModelOverlap.at<int>(cvLabelComps.at<int>(i),projectedIDs.data[i])++;

    if(nMasks)// maskRcnn检测出物体
    {

     // 计算  联通域 和 目标检测mask 重叠度 
        // Compute component-mask overlap
        for (size_t i = 0; i < total; ++i)// total = 480×640
        {
            const unsigned char& mask_val = frame->mask.data[i];// 像素对应的maskid
            const int& comp_val = cvLabelComps.at<int>(i);      // 像素对应的联通域id
            //assert(frame->classIDs.size() > mask_val);
            //if(mask_val != 255)
            compMaskOverlap.at<int>(comp_val,mask_val)++;// 联通域内占用 对应mask id 像素数量
        }

        // Compute mapping
        const float overlap_threshold = 0.65;
        for (int c = 1; c < nComponents; ++c) // 每个联通域
        {
            int& csize = statsComp.at<int>(c, 4);// 该联通域大小
            if(csize > minMappedComponentSize)
            {
                int t = overlap_threshold * csize;
                for (int m = 1; m < nMasks; ++m)
                {
                    if(compMaskOverlap.at<int>(c,m) > t)// 联通域 与 mask 重叠 0.65 以上 
                    {
                        mapComponentToMask[c] = m;// 联通域 c  与 mask m 关联
                        maskComponentPixels[m] += statsComp.at<int>(c, 4);//每个目标检测mask 关联 联通域大小的总面积========  
                                                                          // 一个mask可能关联多个联通域
                                                      // mask 和 多个 联通域关联，所有联通域的矩形框 融合merge
                        maskComponentBoxes[m].mergeLeftTopWidthHeight(statsComp.at<int>(c, 0),
                                                                      statsComp.at<int>(c, 1),
                                                                      statsComp.at<int>(c, 2),
                                                                      statsComp.at<int>(c, 3));
                    }
                }
            }
          else 
          {
                // Map tiny component to ignored
                //mapComponentToMask[c] = 255;

                // Map tiny component to background
                mapComponentToMask[c] = 0;// 联通域太小，不关联 mask，或者就是背景...
            }
        }
    }

    // Replace edges and persons with 255
//    mapComponentToMask[0] = 255; // Edges

    // Group components that belong to the same mask
    for (size_t i = 0; i < total; ++i)
        result.fullSegmentation.data[i] = mapComponentToMask[cvLabelComps.at<int>(i)];// 联通域所属 mask
    TOCK("segmentation-assign");

    // FIX HACK
    for(size_t i=0; i<total; i++)
        if(semanticIgnoreMap.data[i])
            result.fullSegmentation.data[i] = 255;

    if(removeEdgeIslands && nMasks)
    {
        // Remove "edge islands" within masks
      // 阈值滤波===============================
        cv::threshold(result.fullSegmentation, cv8UC1Buffer, 254, 255, cv::THRESH_TOZERO); // THRESH_BINARY is equivalent here
        cv::Mat statsEdgeComp, centroidsEdgeComp;
      
      
      // 又计算了一次联通域====!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        int nEdgeComp = cv::connectedComponentsWithStats(cv8UC1Buffer, cvLabelEdges, statsEdgeComp, centroidsEdgeComp, 4);
        //cv::imshow("edge labels", mapLabelToColorImage(cvLabelEdges));

#ifdef SHOW_DEBUG_VISUALISATION
        cv::Mat islands(height, width, CV_8UC1, cv::Scalar(0));
#endif

        for (int ec = 1; ec < nEdgeComp; ++ec) // 新的联通域数量========================
        {
            for (int m = 1; m < nMasks; ++m)   // 每个mask ===========================
            {
              // 联通域 边框 转换成 边框数据
                BoundingBox bb = BoundingBox::fromLeftTopWidthHeight(statsEdgeComp.at<int>(ec,0),
                                                                     statsEdgeComp.at<int>(ec,1),
                                                                     statsEdgeComp.at<int>(ec,2),
                                                                     statsEdgeComp.at<int>(ec,3));
                if(maskComponentBoxes[m].includes(bb))// mask 的 边框 包含 联通域的边框===================
                {
                    //std::cout << "mask " << m << " fully contains edge-component " << ec << std::endl;
                    int x1 = std::max(bb.left+1,1);           //  边框限制 范围=======
                    int x2 = std::min(bb.right, width-2);
                    int y1 = std::max(bb.top+1, 1);
                    int y2 = std::min(bb.bottom, height-2);
                    bool doBreak = false;
                    for (int y = y1; y <= y2; ++y) // 遍例联通域 边框=============
                    {
                        for (int x = x1; x <= x2; ++x) 
                        {
                            const int& le = cvLabelEdges.at<int>(y,x-1); // 左边 点 对应联通域属性 id
                            const int& te = cvLabelEdges.at<int>(y-1,x); // 上边 点 对应联通域属性 id
                            const int& ce = cvLabelEdges.at<int>(y,x);   // 中间 点 对应联通域属性 id
                          // 对应mask id
                            const unsigned char& lm = result.fullSegmentation.at<unsigned char>(y,x-1);
                            const unsigned char& tm = result.fullSegmentation.at<unsigned char>(y-1,x);
                            const unsigned char& cm = result.fullSegmentation.at<unsigned char>(y,x);
                            if( (le!=ec && ce==ec && lm!=m) ||
                                    (le==ec && ce!=ec && cm!=m) ||
                                    (te!=ec && ce==ec && tm!=m) ||
                                    (te==ec && ce!=ec && cm!=m)) 
                            {
                                doBreak = true;
                                break;
                            }
                        }
                        if(doBreak) break;
                    }
                    if(doBreak) break;

                    // This can only happen once, replace component
                    for (int y = bb.top; y <= bb.bottom; ++y) 
                    {
                        for (int x = bb.left; x <= bb.right; ++x) 
                        {
                            if (cvLabelEdges.at<int>(y,x)==ec)
                            {
                                result.fullSegmentation.at<unsigned char>(y,x) = m;
                                //islands.at<unsigned char>(y,x) = 255;
                            }
                        }
                    }
                }
            }
        }
#ifdef SHOW_DEBUG_VISUALISATION
        cv::imshow("islands", islands);
#endif
    }

    if(nMasks)
    {
        TICK("segmentation-assignModdel");

        // Perform closing on masks
        const int morphElementSize = 2*morphMaskRadius + 1;
      // cv::MORPH_ELLIPSE 椭圆 核
        cv::Mat element = cv::getStructuringElement( cv::MORPH_ELLIPSE, cv::Size(morphElementSize,morphElementSize), cv::Point(morphMaskRadius, morphMaskRadius) );
      // cv::MORPH_CLOSE 闭运算
      cv::morphologyEx(result.fullSegmentation, result.fullSegmentation, cv::MORPH_CLOSE, element, cv::Point(-1,-1), morphMaskIterations);

#ifdef SHOW_DEBUG_VISUALISATION
        cv::imshow( "Before model assignment", mapLabelToColorImage(result.fullSegmentation) );
#endif

        // Try mapping masks to models

        // Init maskToID mapping (background / ignored)
        for (unsigned char midx = 1; midx < nMasks; ++midx) 
        {
            maskToID[midx] = 0;
            if(frame->classIDs[midx]==personClassID) maskToID[midx] = 255; // Person  255为 忽略的
        }

        // Compute overlap with existing models
        for (unsigned char b = 0; b < models.size(); ++b)
            for (int j = 0; j < 256; ++j) modelBuffers[b].maskOverlap[j] = 0; // The compiler will place memset here
        //std::vector<std::vector<unsigned>> maskOverlap(models.size(), {})
        for (size_t i = 0; i < total; ++i) 
        {
            const unsigned char mask = result.fullSegmentation.data[i];// mask id
            for (unsigned char b = 0; b < models.size(); ++b)
                //if (modelBuffers[b].vertConfMap.at<float>(4*i+3) > 0) modelBuffers[b].maskOverlap[mask]++;
                if(projectedIDs.data[i]==modelBuffers[b].modelID) modelBuffers[b].maskOverlap[mask]++;
        }

        // Find best match to model, for each mask
        for (unsigned char midx = 1; midx < nMasks; ++midx) // 每个mask========================
        {
            if(maskToID[midx]==255) continue; // Masks mapped to 255 are ignored
            unsigned char bestModelIndex = 0;
            unsigned int bestOverlap = 0;
            int maskClassID = frame->classIDs[midx];
            for (unsigned char j = 1; j < models.size(); ++j) 
            {
                const unsigned int& overlap = modelBuffers[j].maskOverlap[midx];
//                if(overlap > bestOverlap || (overlap > 10 && bestOverlap==0)){
                if(overlap > bestOverlap)
                {
                    bestOverlap = overlap;
                    bestModelIndex = j;
                }
            }

            bool bestModelMatchesClass = (*result.modelData[bestModelIndex].modelListIterator)->getClassID()==maskClassID;
            if(bestOverlap < minMaskModelOverlap * maskComponentPixels[midx])
            {
                bestModelIndex = 0;
            }

            // Based on match, assign background/existing/new model
            if(bestModelIndex!=0 && bestModelMatchesClass)
            {
                // Assign mask to existing model
                maskToID[midx] = modelBuffers[bestModelIndex].modelID;
                //modelIDToMask[maskToID[midx]] = midx;
                SegmentationResult::ModelData& modelData = result.modelData[bestModelIndex];
                modelData.isEmpty = false;
                modelData.pixelCount = maskComponentPixels[midx];
            } 
          else 
          {
                if(result.hasNewLabel==false && allowNew && 
                   maskComponentPixels[midx] > minNewMaskPixels && 
                   bestModelIndex==0)
                {
                    // Create new model for mask   为 mask 创建新的 3d 模型==================
                    maskToID[midx] = nextModelID;
                    result.hasNewLabel = true;
                    result.modelData.push_back({nextModelID});
                    SegmentationResult::ModelData& md = result.modelData.back();
                    md.isEmpty = false;
                    md.depthMean = 30; // FIXME, this requirement is not intuitive!!
                    md.depthStd = 30;
                    md.classID = maskClassID;
                } 
                else 
                {
                    // Mask is not corresponding to any model
                    maskToID[midx] = 255;
                }
            }
        }
        TOCK("segmentation-assignModdel");
    }

    TICK("segmentation-finalize");
    for (size_t i = 0; i < total; ++i)
        result.fullSegmentation.data[i] = maskToID[result.fullSegmentation.data[i]];

    // Try to map unused components to existing models
    if(true){
        int model, overlap;
        for (int c = 1; c < nComponents; ++c) 
        {
            if(mapComponentToMask[c]==0)
            {
                int overlap = compModelOverlap.row(c).maxCoeff(&model);
                if(model > 0 && overlap > 0.6f * statsComp.at<int>(c, 4)){
                    int x1 = statsComp.at<int>(c, 0);
                    int x2 = statsComp.at<int>(c, 0)+statsComp.at<int>(c, 2);
                    int y1 = statsComp.at<int>(c, 1);
                    int y2 = statsComp.at<int>(c, 1)+statsComp.at<int>(c, 3);
                    for (int y = y1; y <= y2; ++y) 
                    {
                        for (int x = x1; x <= x2; ++x) 
                        {
                            if(cvLabelComps.at<int>(y,x)==c) 
                            {
                                result.fullSegmentation.at<unsigned char>(y,x) = model;
                            }
                        }
                    }
                }
            }
        }
    }


    //cv::imshow("output", overlayMask(frame->rgb, result.fullSegmentation));
    //cv::waitKey(1);

    cudaDeviceSynchronize();
    TOCK("segmentation-finalize");

    cudaCheckError();
#ifdef WRITE_MASK_FILES
    cv::imwrite(WRITE_MASK_DIR + "mrcnn+geom" + std::to_string(frame->index) + ".png", result.fullSegmentation);
    std::cout << "WRITING TO:" << WRITE_MASK_DIR << std::endl;
#endif
    TOCK("segmentation");
    return result;
}

std::vector<std::pair<std::string, std::shared_ptr<GPUTexture> > > MfSegmentation::getDrawableTextures()
{
    return {
        { "BifoldSegmentation", segmentationMap },
        //{ "DebugMap", debugMap }
    };
}

void MfSegmentation::computeLookups(){
    // Copy OpenGL depth texture for CUDA use
    textureDepthMetric->cudaMap();
    cudaArray* depthTexturePtr = textureDepthMetric->getCudaArray();
    cudaMemcpy2DFromArray(depthMapMetric.ptr(0), depthMapMetric.step(), depthTexturePtr, 0, 0, depthMapMetric.colsBytes(), depthMapMetric.rows(),
                          cudaMemcpyDeviceToDevice);
    textureDepthMetric->cudaUnmap();

    textureRGB->cudaMap();
    cudaArray* rgbTexturePtr = textureRGB->getCudaArray();
    cudaMemcpy2DFromArray(rgb.ptr(0), rgb.step(), rgbTexturePtr, 0, 0, rgb.colsBytes(), rgb.rows(), cudaMemcpyDeviceToDevice);
    textureRGB->cudaUnmap();

    // Custom filter for depth map
    bilateralFilter(rgb, depthMapMetric, depthMapMetricFiltered, bilatSigmaRadius, 0, bilatSigmaDepth, bilatSigmaColor, bilatSigmaLocation);
    //    cudaArray* debugMapPtr = debugMap->getCudaArray();
    //    cudaMemcpy2DToArray(debugMapPtr, 0, 0, depthMapMetricFiltered.ptr(0), depthMapMetricFiltered.step(), depthMapMetricFiltered.colsBytes(), depthMapMetricFiltered.rows(), cudaMemcpyDeviceToDevice);

    // Generate buffers for vertex and normal maps
    createVMap(cameraIntrinsics, depthMapMetricFiltered, vertexMap, 999.0f);
    createNMap(vertexMap, normalMap);

    cudaDeviceSynchronize();
    cudaCheckError();
}

void MfSegmentation::allocateModelBuffers(unsigned char numModels)
{
    while(modelBuffers.size() < numModels)
    {
        modelBuffers.emplace_back();
    }
}
