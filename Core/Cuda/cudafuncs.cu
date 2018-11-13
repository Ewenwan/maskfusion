/*
 * This file is part of ElasticFusion.
 * 图像操作 常用函数实现
 * 高斯下采样、深度值三角变换 归一化map 深度转3D点 点转深度 拷贝 变形
 * 2dMAP 间隔 列数行存储 x，y，z值
 * 归一化map: 相邻 三点 构成 两向量 叉乘向量 再归一化
 *  Author: Anatoly Baskeheev, Itseez Ltd, (myname.mysurname@mycompany.com)
 */

#include "cudafuncs.cuh"
#include "convenience.cuh"
#include "operators.cuh"

// 金字塔高斯模糊下采样 函数核 
__global__ void pyrDownGaussKernel (const PtrStepSz<float> src, PtrStepSz<float> dst, float sigma_color)
{
    // 二维线程块
    int x = blockIdx.x * blockDim.x + threadIdx.x;// 按行索引的线程索引
    int y = blockIdx.y * blockDim.y + threadIdx.y;// 按列索引的线程索引

    if (x >= dst.cols || y >= dst.rows)
        return;

    const int D = 5;

    int center = src.ptr (2 * y)[2 * x];

    int x_mi = max(0, 2*x - D/2) - 2*x;
    int y_mi = max(0, 2*y - D/2) - 2*y;

    int x_ma = min(src.cols, 2*x -D/2+D) - 2*x;
    int y_ma = min(src.rows, 2*y -D/2+D) - 2*y;

    float sum = 0;
    float wall = 0;

    float weights[] = {0.375f, 0.25f, 0.0625f} ;

    for(int yi = y_mi; yi < y_ma; ++yi)
        for(int xi = x_mi; xi < x_ma; ++xi)
        {
            int val = src.ptr (2*y + yi)[2*x + xi];

            if (abs (val - center) < 3 * sigma_color)
            {
                sum += val * weights[abs(xi)] * weights[abs(yi)];
                wall += weights[abs(xi)] * weights[abs(yi)];
            }
        }


    dst.ptr (y)[x] = static_cast<int>(sum / wall);
}

// 金字塔下采样函数=============================================================================
void pyrDown(const DeviceArray2D<unsigned short> & src, DeviceArray2D<unsigned short> & dst)
{
    // src 目标图像      gpu二维数组
    // dst 下采样后的图像 
    dst.create (src.rows () / 2, src.cols () / 2);// 下采样后的尺寸
//  1. dim3是基亍uint3定义的矢量类型，相当亍由3个unsigned int型组成的结构体。
//       uint3类型有三个数据成员unsigned int x; unsigned int y; unsigned int z;
//  2. 可使用亍一维、二维或三维的索引来标识线程，构成一维、二维或三维线程块。
//  3. dim3结构类型变量用在核函数调用的<<<,>>>中。
// https://github.com/Ewenwan/ShiYanLou/blob/master/CUDA/readme.md
    
    dim3 block (32, 8);// 二维线程块
    // 线程格
    dim3 grid (getGridDim (dst.cols (), block.x), getGridDim (dst.rows (), block.y));

    const float sigma_color = 30;

    pyrDownGaussKernel<<<grid, block>>>(src, dst, sigma_color);
    cudaCheckError();
}

// 深度值 + 相机内参数计算 x,y,z三维点 核函数==========================================
// Generate a vertex map 'vmap' based on the depth map 'depth' and camera parameters
__global__ void computeVmapKernel(const PtrStepSz<float> depth, 
                                  PtrStep<float> vmap, 
                                  float fx_inv, 
                                  float fy_inv, 
                                  float cx, 
                                  float cy, 
                                  float depthCutoff)
{
    int u = threadIdx.x + blockIdx.x * blockDim.x;
    int v = threadIdx.y + blockIdx.y * blockDim.y;

    if(u < depth.cols && v < depth.rows)
    {
        float z = depth.ptr (v)[u];// 深度值=============

        if(z > 0.0f && z < depthCutoff)
        {
            float vx = z * (u - cx) * fx_inv; // x，y,z三坐标值
            float vy = z * (v - cy) * fy_inv;
            float vz = z;
            // 间隔 列数行存储 x，y，z值
            vmap.ptr (v                 )[u] = vx;// 第一行存 x
            vmap.ptr (v + depth.rows    )[u] = vy;// 第二行(与第一行隔开列数个行)存 y
            vmap.ptr (v + depth.rows * 2)[u] = vz;// 第三行(与第二行隔开列数个行)存 z
        }
        else
        {
            vmap.ptr (v)[u] = __int_as_float(0x7fffffff); /*CUDART_NAN_F*/
            vmap.ptr (v + depth.rows * 2)[u] = 0;
        }
    }
}

// 深度值 + 相机内参数计算 x,y,z三维点 三角变换=================================================
void createVMap(const CameraModel& intr, 
                const DeviceArray2D<float> & depth, 
                DeviceArray2D<float> & vmap, 
                const float depthCutoff)
{
    vmap.create (depth.rows () * 3, depth.cols ());// 一行变3行，存x,y,z三个值

    dim3 block (32, 8);// 二维 线程块
    dim3 grid (1, 1, 1);// 二维线程 格
    grid.x = getGridDim (depth.cols (), block.x);
    grid.y = getGridDim (depth.rows (), block.y);
    // 相机内参数
    float fx = intr.fx, cx = intr.cx;
    float fy = intr.fy, cy = intr.cy;

    computeVmapKernel<<<grid, block>>>(depth, vmap, 1.f / fx, 1.f / fy, cx, cy, depthCutoff);
    cudaSafeCall (cudaGetLastError ());
}

// 归一化map=====相邻三点构成两向量叉乘向量再归一化==========
__global__ void computeNmapKernel(int rows, int cols, const PtrStep<float> vmap, PtrStep<float> nmap)
{
    int u = threadIdx.x + blockIdx.x * blockDim.x;
    int v = threadIdx.y + blockIdx.y * blockDim.y;

    if (u >= cols || v >= rows)
        return;

    if (u == cols - 1 || v == rows - 1)
    {
        nmap.ptr (v)[u] = __int_as_float(0x7fffffff); /*CUDART_NAN_F*/
        return;
    }

    float3 v00, v01, v10;
    // x
    v00.x = vmap.ptr (v  )[u];      //  [V,U]  [V,U+1]
    v01.x = vmap.ptr (v  )[u + 1];  //  [V+1,U] 
    v10.x = vmap.ptr (v + 1)[u];

    if (!isnan (v00.x) && !isnan (v01.x) && !isnan (v10.x))// 右边点，下边点的x值都存在(不是噪点...)
    {
        // 对应隔列书行存储的为 y值
        v00.y = vmap.ptr (v + rows)[u];
        v01.y = vmap.ptr (v + rows)[u + 1];// 右边点
        v10.y = vmap.ptr (v + 1 + rows)[u];// 下边点
        
        // z
        v00.z = vmap.ptr (v + 2 * rows)[u];
        v01.z = vmap.ptr (v + 2 * rows)[u + 1];
        v10.z = vmap.ptr (v + 1 + 2 * rows)[u];

        float3 r = normalized (cross (v01 - v00, v10 - v00));// 相邻 三点 构成 两向量 叉乘向量 再归一化

        nmap.ptr (v       )[u] = r.x;
        nmap.ptr (v + rows)[u] = r.y;
        nmap.ptr (v + 2 * rows)[u] = r.z;
    }
    else
        nmap.ptr (v)[u] = __int_as_float(0x7fffffff); /*CUDART_NAN_F*/
}
// 归一化map===============
void createNMap(const DeviceArray2D<float>& vmap, 
                DeviceArray2D<float>& nmap)
{
    nmap.create (vmap.rows (), vmap.cols ());

    int rows = vmap.rows () / 3;// 输入的map的 行数是扩大了3倍的，一行深度变3行X，Y，Z
    int cols = vmap.cols ();

    dim3 block (32, 8);
    dim3 grid (1, 1, 1);
    grid.x = getGridDim (cols, block.x);
    grid.y = getGridDim (rows, block.y);

    computeNmapKernel<<<grid, block>>>(rows, cols, vmap, nmap);
    cudaSafeCall (cudaGetLastError ());
}




__global__ void tranformMapsKernel(int rows, int cols, 
                                   const PtrStep<float> vmap_src, 
                                   const PtrStep<float> nmap_src,
                                   const mat33 Rmat, 
                                   const float3 tvec, PtrStepSz<float> vmap_dst,
                                   PtrStep<float> nmap_dst)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < cols && y < rows)
    {
        //vertexes
        float3 vsrc, vdst = make_float3 (__int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff));
        vsrc.x = vmap_src.ptr (y)[x];

        if (!isnan (vsrc.x))
        {
            vsrc.y = vmap_src.ptr (y + rows)[x];
            vsrc.z = vmap_src.ptr (y + 2 * rows)[x];

            vdst = Rmat * vsrc + tvec;// 3D点刚体变换  

            vmap_dst.ptr (y + rows)[x] = vdst.y;
            vmap_dst.ptr (y + 2 * rows)[x] = vdst.z;
        }

        vmap_dst.ptr (y)[x] = vdst.x;

        //normals
        float3 nsrc, ndst = make_float3 (__int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff));
        nsrc.x = nmap_src.ptr (y)[x];

        if (!isnan (nsrc.x))
        {
            nsrc.y = nmap_src.ptr (y + rows)[x];
            nsrc.z = nmap_src.ptr (y + 2 * rows)[x];

            ndst = Rmat * nsrc;// 归一化MAP只需要 旋转变换即可！！！！！！！！

            nmap_dst.ptr (y + rows)[x] = ndst.y;
            nmap_dst.ptr (y + 2 * rows)[x] = ndst.z;
        }

        nmap_dst.ptr (y)[x] = ndst.x;
    }
}
// 3Di点 RT刚体变换  ===========================================
void tranformMaps(const DeviceArray2D<float>& vmap_src,
                  const DeviceArray2D<float>& nmap_src,
                  const mat33& Rmat, const float3& tvec,
                  DeviceArray2D<float>& vmap_dst, DeviceArray2D<float>& nmap_dst)
{
    int cols = vmap_src.cols();
    int rows = vmap_src.rows() / 3;

    vmap_dst.create(rows * 3, cols);
    nmap_dst.create(rows * 3, cols);

    dim3 block(32, 8);
    dim3 grid(1, 1, 1);
    grid.x = getGridDim(cols, block.x);
    grid.y = getGridDim(rows, block.y);

    tranformMapsKernel<<<grid, block>>>(rows, cols, vmap_src, nmap_src, Rmat, tvec, vmap_dst, nmap_dst);
    cudaSafeCall(cudaGetLastError());
}



// 拷贝 3D地图点和归一化MAP===========================
__global__ void copyMapsKernel(int rows, int cols, 
                               const float * vmap_src, 
                               const float * nmap_src,
                               PtrStepSz<float> vmap_dst, 
                               PtrStep<float> nmap_dst)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < cols && y < rows)
    {
        //vertexes
        float3 vsrc, vdst = make_float3 (__int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff));

        vsrc.x = vmap_src[y * cols * 4 + (x * 4) + 0];
        vsrc.y = vmap_src[y * cols * 4 + (x * 4) + 1];
        vsrc.z = vmap_src[y * cols * 4 + (x * 4) + 2];

        if(!(vsrc.z == 0))
        {
            vdst = vsrc;
        }

        vmap_dst.ptr (y)[x] = vdst.x;
        vmap_dst.ptr (y + rows)[x] = vdst.y;
        vmap_dst.ptr (y + 2 * rows)[x] = vdst.z;

        //normals
        float3 nsrc, ndst = make_float3 (__int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff));

        nsrc.x = nmap_src[y * cols * 4 + (x * 4) + 0];
        nsrc.y = nmap_src[y * cols * 4 + (x * 4) + 1];
        nsrc.z = nmap_src[y * cols * 4 + (x * 4) + 2];

        if(!(vsrc.z == 0))
        {
            ndst = nsrc;
        }

        nmap_dst.ptr (y)[x] = ndst.x;
        nmap_dst.ptr (y + rows)[x] = ndst.y;
        nmap_dst.ptr (y + 2 * rows)[x] = ndst.z;
    }
}
// 拷贝 3D地图点和归一化MAP========================================
void copyMaps(const DeviceArray<float>& vmap_src,
              const DeviceArray<float>& nmap_src,
              DeviceArray2D<float>& vmap_dst,
              DeviceArray2D<float>& nmap_dst)
{
    int cols = vmap_dst.cols();
    int rows = vmap_dst.rows() / 3;

    vmap_dst.create(rows * 3, cols);
    nmap_dst.create(rows * 3, cols);

    dim3 block(32, 8);
    dim3 grid(1, 1, 1);
    grid.x = getGridDim(cols, block.x);
    grid.y = getGridDim(rows, block.y);

    copyMapsKernel<<<grid, block>>>(rows, cols, vmap_src, nmap_src, vmap_dst, nmap_dst);
    cudaSafeCall(cudaGetLastError());
}



// 自定义高斯核下采样====================================
__global__ void pyrDownKernelGaussF(const PtrStepSz<float> src, 
                                    PtrStepSz<float> dst, 
                                    float * gaussKernel)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst.cols || y >= dst.rows)
        return;

    const int D = 5;

    float center = src.ptr (2 * y)[2 * x];

    int tx = min (2 * x - D / 2 + D, src.cols - 1);
    int ty = min (2 * y - D / 2 + D, src.rows - 1);
    int cy = max (0, 2 * y - D / 2);

    float sum = 0;
    int count = 0;

    for (; cy < ty; ++cy)
    {
        for (int cx = max (0, 2 * x - D / 2); cx < tx; ++cx)
        {
            if(!isnan(src.ptr (cy)[cx]))
            {
                sum += src.ptr (cy)[cx] * gaussKernel[(ty - cy - 1) * 5 + (tx - cx - 1)];
                count += gaussKernel[(ty - cy - 1) * 5 + (tx - cx - 1)];
            }
        }
    }
    dst.ptr (y)[x] = (float)(sum / (float)count);
}
// 5×5高斯核下采样==float类型======================
void pyrDownGaussF(const DeviceArray2D<float>& src, 
                   DeviceArray2D<float> & dst)
{
    dst.create (src.rows () / 2, src.cols () / 2);

    dim3 block (32, 8);
    dim3 grid (getGridDim (dst.cols (), block.x), getGridDim (dst.rows (), block.y));

    const float gaussKernel[25] = 
                   {1, 4, 6, 4, 1,
                    4, 16, 24, 16, 4,
                    6, 24, 36, 24, 6,
                    4, 16, 24, 16, 4,
                    1, 4, 6, 4, 1};

    float * gauss_cuda;

    cudaSafeCall(cudaMalloc((void**) &gauss_cuda, sizeof(float) * 25));
    cudaSafeCall(cudaMemcpy(gauss_cuda, &gaussKernel[0], sizeof(float) * 25, cudaMemcpyHostToDevice));

    pyrDownKernelGaussF<<<grid, block>>>(src, dst, gauss_cuda);
    cudaCheckError();

    cudaFree(gauss_cuda);
}


// map变形==================================
template<bool normalize>
__global__ void resizeMapKernel(int drows, int dcols, int srows, 
                                const PtrStep<float> input, 
                                PtrStep<float> output)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= dcols || y >= drows)
        return;

    const float qnan = __int_as_float(0x7fffffff);

    int xs = x * 2;
    int ys = y * 2;

    float x00 = input.ptr (ys + 0)[xs + 0];
    float x01 = input.ptr (ys + 0)[xs + 1];
    float x10 = input.ptr (ys + 1)[xs + 0];
    float x11 = input.ptr (ys + 1)[xs + 1];

    if (isnan (x00) || isnan (x01) || isnan (x10) || isnan (x11))
    {
        output.ptr (y)[x] = qnan;
        return;
    }
    else
    {
        float3 n;

        n.x = (x00 + x01 + x10 + x11) / 4;

        float y00 = input.ptr (ys + srows + 0)[xs + 0];
        float y01 = input.ptr (ys + srows + 0)[xs + 1];
        float y10 = input.ptr (ys + srows + 1)[xs + 0];
        float y11 = input.ptr (ys + srows + 1)[xs + 1];

        n.y = (y00 + y01 + y10 + y11) / 4;

        float z00 = input.ptr (ys + 2 * srows + 0)[xs + 0];
        float z01 = input.ptr (ys + 2 * srows + 0)[xs + 1];
        float z10 = input.ptr (ys + 2 * srows + 1)[xs + 0];
        float z11 = input.ptr (ys + 2 * srows + 1)[xs + 1];

        n.z = (z00 + z01 + z10 + z11) / 4;

        if (normalize)
            n = normalized (n);

        output.ptr (y        )[x] = n.x;
        output.ptr (y + drows)[x] = n.y;
        output.ptr (y + 2 * drows)[x] = n.z;
    }
}
// map变形==================================
template<bool normalize>
void resizeMap(const DeviceArray2D<float>& input, DeviceArray2D<float>& output)
{
    int in_cols = input.cols ();
    int in_rows = input.rows () / 3;

    int out_cols = in_cols / 2;
    int out_rows = in_rows / 2;

    output.create (out_rows * 3, out_cols);

    dim3 block (32, 8);
    dim3 grid (getGridDim (out_cols, block.x), getGridDim (out_rows, block.y));
    resizeMapKernel<normalize><< < grid, block>>>(out_rows, out_cols, in_rows, input, output);
    cudaCheckError();
    cudaSafeCall (cudaDeviceSynchronize ());
}

void resizeVMap(const DeviceArray2D<float>& input, DeviceArray2D<float>& output)
{
    resizeMap<false>(input, output);
}

void resizeNMap(const DeviceArray2D<float>& input, DeviceArray2D<float>& output)
{
    resizeMap<true>(input, output);
}

//FIXME Remove
/*
void launch_kernel(float4 *pos, unsigned int mesh_width,
                   unsigned int mesh_height, float time)
{
    // execute the kernel
    dim3 block(8, 8, 1);
    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
    simple_vbo_kernel<<< grid, block>>>(pos, mesh_width, mesh_height, time);
}*/

//FIXME Remove
/*
__global__ void testKernel(cudaSurfaceObject_t tex)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 960 || y >= 540)
        return;

    / *
    const int D = 5;

    float center = src.ptr (2 * y)[2 * x];

    int tx = min (2 * x - D / 2 + D, src.cols - 1);
    int ty = min (2 * y - D / 2 + D, src.rows - 1);
    int cy = max (0, 2 * y - D / 2);

    float sum = 0;
    int count = 0;

    for (; cy < ty; ++cy)
    {
        for (int cx = max (0, 2 * x - D / 2); cx < tx; ++cx)
        {
            if(!isnan(src.ptr (cy)[cx]))
            {
                sum += src.ptr (cy)[cx] * gaussKernel[(ty - cy - 1) * 5 + (tx - cx - 1)];
                count += gaussKernel[(ty - cy - 1) * 5 + (tx - cx - 1)];
            }
        }
    }* /
    //dst.ptr (y)[x] = (float)(sum / (float)count);
    //data[y * 960 + x] = x / 960.0;
    //data[8] = 0.4;
    float1 test = make_float1(0.99);
    surf2Dwrite(test, tex, x*sizeof(float1), y);
}

//FIXME Remove
void testCuda(cudaSurfaceObject_t surface)//(float* data)
{
    //dst.create (src.rows () / 2, src.cols () / 2);

    dim3 block (32, 8);
    dim3 grid (getGridDim (960, block.x), getGridDim (540, block.y));

    testKernel<<<grid, block>>>(surface);
    cudaCheckError();
}*/


// 5×5高斯核下采样==unsigned char类型======================
__global__ void pyrDownKernelIntensityGauss(const PtrStepSz<unsigned char> src, 
                                            PtrStepSz<unsigned char> dst, 
                                            float * gaussKernel)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst.cols || y >= dst.rows)
        return;

    const int D = 5;

    int center = src.ptr (2 * y)[2 * x];

    int tx = min (2 * x - D / 2 + D, src.cols - 1);
    int ty = min (2 * y - D / 2 + D, src.rows - 1);
    int cy = max (0, 2 * y - D / 2);

    float sum = 0;
    int count = 0;

    for (; cy < ty; ++cy)
        for (int cx = max (0, 2 * x - D / 2); cx < tx; ++cx)
        {
            //This might not be right, but it stops incomplete model images from making up colors
            if(src.ptr (cy)[cx] > 0)
            {
                sum += src.ptr (cy)[cx] * gaussKernel[(ty - cy - 1) * 5 + (tx - cx - 1)];
                count += gaussKernel[(ty - cy - 1) * 5 + (tx - cx - 1)];
            }
        }
    dst.ptr (y)[x] = (sum / (float)count);
}
// 5×5高斯核下采样==unsigned char类型======================
void pyrDownUcharGauss(const DeviceArray2D<unsigned char>& src, 
                       DeviceArray2D<unsigned char> & dst)
{
    dst.create (src.rows () / 2, src.cols () / 2);

    dim3 block (32, 8);
    dim3 grid (getGridDim (dst.cols (), block.x), getGridDim (dst.rows (), block.y));
    
    //  5×5高斯核
    const float gaussKernel[25] = {1, 4, 6, 4, 1,
                    4, 16, 24, 16, 4,
                    6, 24, 36, 24, 6,
                    4, 16, 24, 16, 4,
                    1, 4, 6, 4, 1};

    float * gauss_cuda;

    cudaMalloc((void**) &gauss_cuda, sizeof(float) * 25);
    cudaMemcpy(gauss_cuda, &gaussKernel[0], sizeof(float) * 25, cudaMemcpyHostToDevice);

    pyrDownKernelIntensityGauss<<<grid, block>>>(src, dst, gauss_cuda);
    cudaCheckError();

    cudaFree(gauss_cuda);
}

/*void pyrDown2(const DeviceArray2D<unsigned char> & src, DeviceArray2D<unsigned char> & dst)
{
    dst.create (src.rows () / 2, src.cols () / 2);

    dim3 block (32, 8);
    dim3 grid (getGridDim (dst.cols (), block.x), getGridDim (dst.rows (), block.y));

    //pyrDownUcharGauss<<<grid, block>>>(src, dst);
    pyrDownUcharGauss()
    cudaCheckError();
}*/

// 3d点转 深度图===================================
__global__ void verticesToDepthKernel(const float * vmap_src, PtrStepSz<float> dst, float cutOff)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst.cols || y >= dst.rows)
        return;

    float z = vmap_src[y * dst.cols * 4 + (x * 4) + 2];

    dst.ptr(y)[x] = z > cutOff || z <= 0 ? __int_as_float(0x7fffffff)/*CUDART_NAN_F*/ : z;
}
// 3d map转 深度图===================================
void verticesToDepth(DeviceArray<float>& vmap_src, DeviceArray2D<float> & dst, float cutOff)
{
    dim3 block (32, 8);
    dim3 grid (getGridDim (dst.cols (), block.x), getGridDim (dst.rows (), block.y));

    verticesToDepthKernel<<<grid, block>>>(vmap_src, dst, cutOff);
    cudaCheckError();
}

texture<uchar4, 2, cudaReadModeElementType> inTex;

// RGB图转灰度图=======================================
__global__ void bgr2IntensityKernel(PtrStepSz<unsigned char> dst)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst.cols || y >= dst.rows)
        return;

    uchar4 src = tex2D(inTex, x, y);
//  Gray = R*0.299 + G*0.587 + B*0.114; 
//  Gray = (R*299 + G*587 + B*114 + 500); // 1000所以需要加上500来实现四舍五入。
//  Gray = (R*30 + G*59 + B*11 + 50) / 100; 
//  Gray = (R*38 + G*75 + B*15) >> 7
//  Gray = (R + (WORD)G<<1 + B) >> 2
    int value = (float)src.x * 0.114f + (float)src.y * 0.299f + (float)src.z * 0.587f;

    dst.ptr (y)[x] = value;
}
// RGB图转灰度图=======================================
void imageBGRToIntensity(cudaArray * cuArr, DeviceArray2D<unsigned char> & dst)
{
    dim3 block (32, 8);
    dim3 grid (getGridDim (dst.cols (), block.x), getGridDim (dst.rows (), block.y));

    cudaSafeCall(cudaBindTextureToArray(inTex, cuArr));

    bgr2IntensityKernel<<<grid, block>>>(dst);

    cudaCheckError();

    cudaSafeCall(cudaUnbindTexture(inTex));
}

__constant__ float gsobel_x3x3[9];
__constant__ float gsobel_y3x3[9];

__global__ void applyKernel(const PtrStepSz<unsigned char> src, 
                            PtrStep<short> dx, 
                            PtrStep<short> dy)
{

  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if(x >= src.cols || y >= src.rows)
    return;

  float dxVal = 0;
  float dyVal = 0;

  int kernelIndex = 8;
  for(int j = max(y - 1, 0); j <= min(y + 1, src.rows - 1); j++)
  {
      for(int i = max(x - 1, 0); i <= min(x + 1, src.cols - 1); i++)
      {
          dxVal += (float)src.ptr(j)[i] * gsobel_x3x3[kernelIndex];
          dyVal += (float)src.ptr(j)[i] * gsobel_y3x3[kernelIndex];
          --kernelIndex;
      }
  }

  dx.ptr(y)[x] = dxVal;
  dy.ptr(y)[x] = dyVal;
}

// 计算图像梯度====================================
void computeDerivativeImages(DeviceArray2D<unsigned char>& src, DeviceArray2D<short>& dx, DeviceArray2D<short>& dy)
{
    static bool once = false;

    if(!once)
    {
        // x 水平方向 导数模板 
        float gsx3x3[9] = {0.52201,  0.00000, -0.52201,
                           0.79451, -0.00000, -0.79451,
                           0.52201,  0.00000, -0.52201};
        // y 垂直方向 导数模板
        float gsy3x3[9] = {0.52201, 0.79451, 0.52201,
                           0.00000, 0.00000, 0.00000,
                           -0.52201, -0.79451, -0.52201};

        cudaMemcpyToSymbol(gsobel_x3x3, gsx3x3, sizeof(float) * 9);
        cudaMemcpyToSymbol(gsobel_y3x3, gsy3x3, sizeof(float) * 9);

        cudaSafeCall(cudaGetLastError());
        cudaSafeCall(cudaDeviceSynchronize());

        once = true;
    }

    dim3 block(32, 8);
    dim3 grid(getGridDim (src.cols (), block.x), getGridDim (src.rows (), block.y));

    applyKernel<<<grid, block>>>(src, dx, dy);

    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());
}


// 
__global__ void projectPointsKernel(const PtrStepSz<float> depth,
                                    PtrStepSz<float3> cloud,
                                    const float invFx,
                                    const float invFy,
                                    const float cx,
                                    const float cy)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= depth.cols || y >= depth.rows)
        return;

    float z = depth.ptr(y)[x];

    cloud.ptr(y)[x].x = (float)((x - cx) * z * invFx);
    cloud.ptr(y)[x].y = (float)((y - cy) * z * invFy);
    cloud.ptr(y)[x].z = z;
}

// 深度图 转 点云， 维度不变，每个元素是一个三维变量，分别存储x，y，z
void projectToPointCloud(const DeviceArray2D<float> & depth,
                         const DeviceArray2D<float3> & cloud,
                         CameraModel & intrinsics,
                         const int & level)
{
    dim3 block (32, 8);
    dim3 grid (getGridDim (depth.cols (), block.x), getGridDim (depth.rows (), block.y));

    CameraModel intrinsicsLevel = intrinsics(level);

    projectPointsKernel<<<grid, block>>>(depth, cloud, 1.0f / intrinsicsLevel.fx, 1.0f / intrinsicsLevel.fy, intrinsicsLevel.cx, intrinsicsLevel.cy);
    cudaCheckError();
    cudaSafeCall (cudaDeviceSynchronize ());
}
