# Datawhale 计算机视觉基础-图像处理（上）-Task04 图像滤波

## 4.1 简介 
图像的实质是一种二维信号，滤波是信号处理中的一个重要概念。在图像处理中，滤波是一种非常常见的技术，它们的原理非常简单，但是其思想却十分值得借鉴，滤波是很多图像算法的前置步骤或基础，掌握图像滤波对理解卷积神经网络也有一定帮助。


## 4.2 学习目标

* 了解图像滤波的分类和基本概念

* 理解均值滤波/方框滤波、高斯滤波的原理

* 掌握OpenCV框架下滤波API的使用

## 4.3 内容介绍
1、均值滤波/方框滤波、高斯滤波的原理

2、OpenCV代码实践

3、动手实践并打卡（读者完成）

## 4.4 算法理论介绍
### 4.4.1 均值滤波、方框滤波
**1. 滤波分类**

**线性滤波：** 对邻域中的像素的计算为线性运算时，如利用窗口函数进行平滑加权求和的运算，或者某种卷积运算，都可以称为线性滤波。

图像滤波的目的有两个:一是抽出对象的特征作为图像识别的特征模式;另一个是为适应图像处理的要求，消除图像数字化时所混入的噪声。

而对滤波处理的要求也有两条:一是不能损坏图像的轮廓及边缘等重要信息;二是使图像清晰视觉效果好。

平滑滤波是低频增强的空间域滤波技术。它的目的有两类：一类是模糊；另一类是消除噪音。
空间域的平滑滤波一般采用简单平均法进行，就是求邻近像元点的平均亮度值。邻域的大小与平滑的效果直接相关，邻域越大平滑的效果越好，但邻域过大，平滑会使边缘信息损失的越大，从而使输出的图像变得模糊，因此需合理选择邻域的大小。

常见的线性滤波有：均值滤波、高斯滤波、盒子滤波、拉普拉斯滤波等等，通常线性滤波器之间只是模版系数不同。

**非线性滤波：** 非线性滤波利用原始图像跟模版之间的一种逻辑关系得到结果，关于滤波器，一种形象的比喻法是：我们可以把滤波器想象成一个包含加权系数的窗口，当使用这个滤波器平滑处理图像时，就把这个窗口放到图像之上，透过这个窗口来看我们得到的图像。如最值滤波器，中值滤波器。比较常用的有中值滤波器和双边滤波器。、

**2. 邻域算子与线性邻域滤波**

邻域算子（局部算子）是利用给定像素周围的像素值的决定此像素的最终输出值的一种算子。而线性邻域滤波是一种常用的邻域算子，像素的输出值取决于输入像素的加权和，具体过程如下图。

邻域算子除了用于局部色调调整以外，还可以用于图像滤波，实现图像的平滑和锐化，图像边缘增强或者图像噪声的去除。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200428005917702.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQ0MzE1OTg3,size_16,color_FFFFFF,t_70)

上图蓝色点可以解释为：

$$6.5 + 9.8 + 12.3 + 6.5 + 19.2 + 11.5 + 6.3 + 9.1 + 10.7=91.9 \approx 92$$

如上图可知：一个$8*8$的矩阵被一个$3*3$的算子扫完后的形状是$(8-3+1) * (8-3+1)$即$6*6$的

线性滤波处理的输出像素值$g(i,j)$是输入像素值$f(i+k,j+l)$的加权和 :

$$g(i, j)=\sum_{k,l}f(i+k, j+l)h(k, l)$$

其中的加权和为 ，我们称其为“核”，滤波器的加权系数，即滤波器的“滤波系数”。
上面的式子可以简单写作：
 
$$g=f\otimes h$$


其中f表示输入像素值，h表示加权系数“核“，g表示输出像素值




**3. 方框（盒子）滤波**

方框滤波是一种非常有用的线性滤波，也叫盒子滤波，均值滤波就是盒子滤波归一化的特殊情况。
**应用：** 可以说，一切需要求某个邻域内像素之和的场合，都有方框滤波的用武之地，比如：均值滤波、引导滤波、计算Haar特征等等。

**优势：** 就一个字：快！它可以使复杂度为O(MN)的求和，求方差等运算降低到O(1)或近似于O(1)的复杂度，也就是说与邻域尺寸无关了，有点类似积分图吧，但是比积分图更快（与它的实现方式有关）。

在原理上，是采用一个卷积核与图像进行卷积：
      
<div align=center><img width = '220' height ='100' src ="https://img-blog.csdnimg.cn/20200412125446419.png"/></div>

其中：
   <div align=center><img width = '400' height ='40' src ="https://img-blog.csdnimg.cn/20200412125657652.png"/></div>                  

其中ksize.width、ksize.height是核宽度和长度而不是图片的宽度、长度。可见，归一化了就是均值滤波；不归一化则可以计算每个像素邻域上的各种积分特性，方差、协方差，平方和等等。

**4. 均值滤波**

**均值滤波的应用场合：**
根据冈萨雷斯书中的描述，均值模糊可以模糊图像以便得到感兴趣物体的粗略描述，也就是说，去除图像中的不相关细节，其中“不相关”是指与滤波器模板尺寸相比较小的像素区域，从而对图像有一个整体的认知。即为了对感兴趣的物体得到一个大致的整体的描述而模糊一幅图像，忽略细小的细节。

**均值滤波的缺陷：**
均值滤波本身存在着固有的缺陷，即它不能很好地保护图像细节，在图像去噪的同时也破坏了图像的细节部分，从而使图像变得模糊，不能很好地去除噪声点。特别是椒盐噪声。

均值滤波是上述方框滤波的特殊情况，均值滤波方法是：对待处理的当前像素，选择一个模板，该模板为其邻近的若干个像素组成，用模板的均值（方框滤波归一化）来替代原像素的值。公式表示为：

<div align=center><img   src ="https://img-blog.csdnimg.cn/2020041213054619.png"/></div>           

g(x,y)为该邻域的中心像素，n跟系数模版大小有关，一般3*3邻域的模板，n取为9，如：

<div align=center><img   src ="https://img-blog.csdnimg.cn/20200412130646624.png"/></div>           

当然，模板是可变的，一般取奇数，如5 * 5 , 7 * 7等等。

注：在实际处理过程中可对图像边界进行扩充，扩充为0或扩充为邻近的像素值。

**5.中值滤波**

中值滤波是一种典型的非线性滤波，是基于排序统计理论的一种能够有效抑制噪声的非线性信号处理技术，基本思想是用像素点邻域灰度值的中值来代替该像素点的灰度值，让周围的像素值接近真实的值从而消除孤立的噪声点。该方法在取出脉冲噪声、椒盐噪声的同时能保留图像的边缘细节。这些优良特性是线性滤波所不具备的。

中值滤波首先也得生成一个滤波模板，将该模板内的各像素值进行排序，生成单调上升或单调下降的二维数据序列，二维中值滤波输出为


<div align=center><img   src ="https://img-blog.csdnimg.cn/2020042802014220.png"/></div>    

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200428021512767.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQ0MzE1OTg3,size_16,color_FFFFFF,t_70)



求九个数的中值

```cpp
uchar Median(uchar n1, uchar n2, uchar n3, uchar n4, uchar n5,
	uchar n6, uchar n7, uchar n8, uchar n9) {
	uchar arr[9];
	arr[0] = n1;
	arr[1] = n2;
	arr[2] = n3;
	arr[3] = n4;
	arr[4] = n5;
	arr[5] = n6;
	arr[6] = n7;
	arr[7] = n8;
	arr[8] = n9;
	for (int gap = 9 / 2; gap > 0; gap /= 2)//希尔排序
		for (int i = gap; i < 9; ++i)
			for (int j = i - gap; j >= 0 && arr[j] > arr[j + gap]; j -= gap)
				swap(arr[j], arr[j + gap]);
	return arr[4];//返回中值
}
```

### 4.4.1 高斯滤波
**应用：** 高斯滤波是一种线性平滑滤波器，对于服从正态分布的噪声有很好的抑制作用。在实际场景中，我们通常会假定图像包含的噪声为高斯白噪声，所以在许多实际应用的预处理部分，都会采用高斯滤波抑制噪声，如传统车牌识别等。

高斯滤波和均值滤波一样，都是利用一个掩膜和图像进行卷积求解。不同之处在于：均值滤波器的模板系数都是相同的为1，而高斯滤波器的模板系数，则随着距离模板中心的增大而系数减小（服从二维高斯分布）。所以，高斯滤波器相比于均值滤波器对图像个模糊程度较小，更能够保持图像的整体细节。

二维高斯分布
高斯分布公式终于要出场了！

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200412165241878.png)  
其中不必纠结于系数，因为它只是一个常数！并不会影响互相之间的比例关系，并且最终都要进行归一化，所以在实际计算时我们是忽略它而只计算后半部分:
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200412165426976.png)  
其中(x,y)为掩膜内任一点的坐标，(ux,uy)为掩膜内中心点的坐标，在图像处理中可认为是整数；σ是标准差。

例如：要产生一个3×3的高斯滤波器模板，以模板的中心位置为坐标原点进行取样。模板在各个位置的坐标，如下所示（x轴水平向右，y轴竖直向下）。  

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200412165500464.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MDY0NzgxOQ==,size_16,color_FFFFFF,t_70)  
这样，将各个位置的坐标带入到高斯函数中，得到的值就是模板的系数。
对于窗口模板的大小为 (2k+1)×(2k+1)，模板中各个元素值的计算公式如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200412165519667.png)  
这样计算出来的模板有两种形式：小数和整数。

* 小数形式的模板，就是直接计算得到的值，没有经过任何的处理；
* 整数形式的，则需要进行归一化处理，将模板左上角的值归一化为1，其他的按照比例化为整数。使用整数的模板时，需要在模板的前面加一个系数，系数为模板系数和的倒数。

**生成高斯掩膜（小数形式）**  
知道了高斯分布原理，实现起来也就不困难了。

首先我们要确定我们生产掩模的尺寸wsize，然后设定高斯分布的标准差。生成的过程，我们首先根据模板的大小，找到模板的中心位置center。 然后就是遍历，根据高斯分布的函数，计算模板中每个系数的值。

最后模板的每个系数要除以所有系数的和。这样就得到了小数形式的模板。 

```cpp
///////////////////////////////
//x，y方向联合实现获取高斯模板
//////////////////////////////
void generateGaussMask(cv::Mat& Mask,cv::Size wsize, double sigma){
	Mask.create(wsize,CV_64F);
	int h = wsize.height;
	int w = wsize.width;
	int center_h = (h - 1) / 2;
	int center_w = (w - 1) / 2;
	double sum = 0.0;
	double x, y;
	for (int i = 0; i < h; ++i){
		y = pow(i - center_h, 2);
		for (int j = 0; j < w; ++j){
			x = pow(j - center_w, 2);
			//因为最后都要归一化的，常数部分可以不计算，也减少了运算量
			double g = exp(-(x + y) / (2 * sigma*sigma));
			Mask.at<double>(i, j) = g;
			sum += g;
		}
	}
	Mask = Mask / sum;
}
```
**3×3,σ=0.8的小数型模板：**   

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200412165756162.png)  

**σ的意义及选取**   
通过上述的实现过程，不难发现，高斯滤波器模板的生成最重要的参数就是高斯分布的标准差σ。标准差代表着数据的离散程度，如果σ较小，那么生成的模板的中心系数较大，而周围的系数较小，这样对图像的平滑效果就不是很明显；反之，σ较大，则生成的模板的各个系数相差就不是很大，比较类似均值模板，对图像的平滑效果比较明显。

来看下一维高斯分布的概率分布密度图：
![](https://img-blog.csdnimg.cn/20200412165824372.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MDY0NzgxOQ==,size_16,color_FFFFFF,t_70)
于是我们有如下结论：σ越小分布越瘦高，σ越大分布越矮胖。

* σ越大，分布越分散，各部分比重差别不大，于是生成的模板各元素值差别不大，类似于平均模板；
* σ越小，分布越集中，中间部分所占比重远远高于其他部分，反映到高斯模板上就是中心元素值远远大于其他元素值，于是自然而然就相当于中间值得点运算。
### 4.4.2填充问题
在对图像应用滤波器进行过滤时，边界问题是一个需要处理的问题。一般来说，有3种处理的方法。

1. 不做边界处理
不对图像的边界作任何处理，在对图像进行滤波时，滤波器没有作用到图像的四周，因此图像的四周没有发生改变。

 

2. 填充0
对图像的边界做扩展，在扩展边界中填充0，对于边长为2k+1的方形滤波器，扩展的边界大小为k，若原来的图像为[m, n]，则扩展后图像变为[m+2k, n+2k]。进行滤波之后，图像会出现一条黑色的边框。

 

3. 填充最近像素值
扩展与 填充0 的扩展类似，只不过填充0的扩展是在扩展部分填充0，而这个方法是填充距离最近的像素的值。


## 4.5 基于OpenCV的实现

* 工具：OpenCV3.1.0+VS2013
* 平台：WIN10

### 函数原型（c++）
**1.方框滤波**
```cpp
void boxFilter( InputArray src, OutputArray dst, 
                int ddepth,
                Size ksize,  
                Point anchor = Point(-1,-1),
                bool normalize = true,
                int borderType = BORDER_DEFAULT );
```
参数：
 - src – input image.
 - dst – output image of the same size and type as src. 
 - ddepth – the output image depth (-1 to use src.depth()). 
 - ksize – blurring kernel size. anchor 
 - anchor point; default value Point(-1,-1) means that the anchor is at the kernel center. 
 - normalize – flag, specifying  whether the kernel is normalized by its area or not. 
 - borderType –   border mode used to extrapolate pixels outside of the image. 可参考：[cv::BorderTypes](https://docs.opencv.org/3.1.0/d2/de8/group__core__array.html#ga209f2f4869e304c82d07739337eae7c5)

**2.均值滤波**

```cpp
void cv::blur	(	InputArray 	src,
                    OutputArray dst,
                       Size 	ksize,
                      Point 	anchor = Point(-1,-1),
                       int 	  borderType = BORDER_DEFAULT 
)	
```
参数：

* src	– input image; it can have any number of channels, which are processed independently, but the 
*  depth – should be CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
* dst	– output image of the same size and type as src.
* ksize	– blurring kernel size.
* anchor	–  anchor point; default value Point(-1,-1) means that the anchor is at the kernel center.
* borderType – 	border mode used to extrapolate pixels outside of the image,可参考：[cv::BorderTypes](https://docs.opencv.org/3.1.0/d2/de8/group__core__array.html#ga209f2f4869e304c82d07739337eae7c5)

**2.高斯滤波**

```cpp
void GaussianBlur(InputArray src, OutputArray dst, 
                  Size ksize, 
                  double sigmaX, double sigmaY=0,
                  int borderType=BORDER_DEFAULT )
```
参数：
* src — input image; the image can have any number of channels, which are processed independently, but the depth should be CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
* dst   —  output image of the same size and type as src.
* ksize	Gaussian kernel size. ksize.width and ksize.height can differ but they both must be positive and   odd. Or, they can be zero's and then they are computed from sigma.
* sigmaX	 —  Gaussian kernel standard deviation in X direction.
* sigmaY  — 	Gaussian kernel standard deviation in Y direction; if sigmaY is zero, it is set to be equal to sigmaX, if both sigmas are zeros, they are computed from ksize.width and ksize.height, respectively (see cv::getGaussianKernel for details); to fully control the result regardless of possible future modifications of all this semantics, it is recommended to specify all of ksize, sigmaX, and sigmaY.
* borderType  — 	pixel extrapolation method, 可参考：[cv::BorderTypes](https://docs.opencv.org/3.1.0/d2/de8/group__core__array.html#ga209f2f4869e304c82d07739337eae7c5)
### 实现示例（c++)

```cpp
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
 
using namespace cv;
 
int main()
{
	//载入图像
	Mat image = imread("1.jpg");
	Mat dst1 , dst2，dst3;
	 //均值滤波
	blur(image, dst1, Size(7, 7));
	//方框滤波
	cv::boxFilter(image, dst2, -1, cv::Size(7, 7), cv::Point(-1, -1), true, cv::BORDER_CONSTANT);
	//高斯滤波
	cv:: GaussianBlur(image, dst3,cv::Size(7, 7),0.8);
	
    //创建窗口并显示
	namedWindow("均值滤波效果图");
    namedWindow("方框滤波效果图");
    namedWindow("高斯滤波效果图");
	imshow("均值滤波效果图", dst1);
    imshow("方框滤波效果图", dts2);
    imshow("高斯滤波效果图", dts3);
	waitKey(0);
	return 0;
}
```

### 进阶实现(根据原理自己实现)
* 1.方框滤波
```cpp
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
 
/////////////////////////////////////////
//求积分图-优化方法
//由上方negral(i-1,j)加上当前行的和即可
//对于W*H图像：2*(W-1)*(H-1)次加减法
//比常规方法快1.5倍左右
/////////////////////////////////////////
void Fast_integral(cv::Mat& src, cv::Mat& dst){
	int nr = src.rows;
	int nc = src.cols;
	int sum_r = 0;
	dst = cv::Mat::zeros(nr + 1, nc + 1, CV_64F);
	for (int i = 1; i < dst.rows; ++i){
		for (int j = 1, sum_r = 0; j < dst.cols; ++j){
			//行累加，因为积分图相当于在原图上方加一行，左边加一列，所以积分图的(1,1)对应原图(0,0),(i,j)对应(i-1,j-1)
			sum_r = src.at<uchar>(i - 1, j - 1) + sum_r; //行累加
			dst.at<double>(i, j) = dst.at<double>(i - 1, j) + sum_r;
		}
	}
}
 
//////////////////////////////////
//盒子滤波-均值滤波是其特殊情况
/////////////////////////////////
void BoxFilter(cv::Mat& src, cv::Mat& dst, cv::Size wsize, bool normalize){
 
	//图像边界扩充
	if (wsize.height % 2 == 0 || wsize.width % 2 == 0){
		fprintf(stderr, "Please enter odd size!");
		exit(-1);
	}
	int hh = (wsize.height - 1) / 2;
	int hw = (wsize.width - 1) / 2;
	cv::Mat Newsrc;
	cv::copyMakeBorder(src, Newsrc, hh, hh, hw, hw, cv::BORDER_REFLECT);//以边缘为轴，对称
	src.copyTo(dst);
 
	//计算积分图
	cv::Mat inte;
	Fast_integral(Newsrc, inte);
 
	//BoxFilter
	double mean = 0;
	for (int i = hh + 1; i < src.rows + hh + 1; ++i){  //积分图图像比原图（边界扩充后的）多一行和一列 
		for (int j = hw + 1; j < src.cols + hw + 1; ++j){
			double top_left = inte.at<double>(i - hh - 1, j - hw - 1);
			double top_right = inte.at<double>(i - hh - 1, j + hw);
			double buttom_left = inte.at<double>(i + hh, j - hw - 1);
			double buttom_right = inte.at<double>(i + hh, j + hw);
			if (normalize == true)
				mean = (buttom_right - top_right - buttom_left + top_left) / wsize.area();
			else
				mean = buttom_right - top_right - buttom_left + top_left;
			
			//一定要进行判断和数据类型转换
			if (mean < 0)
				mean = 0;
			else if (mean>255)
				mean = 255;
			dst.at<uchar>(i - hh - 1, j - hw - 1) = static_cast<uchar>(mean);
 
		}
	}
}
 
int main(){
	cv::Mat src = cv::imread("I:\\Learning-and-Practice\\2019Change\\Image process algorithm\\Img\\woman2.jpeg");
	if (src.empty()){
		return -1;
	}
 
	//自编BoxFilter测试
	cv::Mat dst1;
	double t2 = (double)cv::getTickCount(); //测时间
	if (src.channels() > 1){
		std::vector<cv::Mat> channel;
		cv::split(src, channel);
		BoxFilter(channel[0], channel[0], cv::Size(7, 7), true);//盒子滤波
		BoxFilter(channel[1], channel[1], cv::Size(7, 7), true);//盒子滤波
		BoxFilter(channel[2], channel[2], cv::Size(7, 7), true);//盒子滤波
		cv::merge(channel,dst1);
	}else
		BoxFilter(src, dst1, cv::Size(7, 7), true);//盒子滤波
	t2 = (double)cv::getTickCount() - t2;
	double time2 = (t2 *1000.) / ((double)cv::getTickFrequency());
	std::cout << "FASTmy_process=" << time2 << " ms. " << std::endl << std::endl;
 
	//opencv自带BoxFilter测试
	cv::Mat dst2;
	double t1 = (double)cv::getTickCount(); //测时间
	cv::boxFilter(src, dst2, -1, cv::Size(7, 7), cv::Point(-1, -1), true, cv::BORDER_CONSTANT);//盒子滤波
	t1 = (double)cv::getTickCount() - t1;
	double time1 = (t1 *1000.) / ((double)cv::getTickFrequency());
	std::cout << "Opencvbox_process=" << time1 << " ms. " << std::endl << std::endl;
 
	cv::namedWindow("src");
	cv::imshow("src", src);
	cv::namedWindow("ourdst",CV_WINDOW_NORMAL);
	cv::imshow("ourdst", dst1);
	cv::namedWindow("opencvdst", CV_WINDOW_NORMAL);
	cv::imshow("opencvdst", dst2);
	cv::waitKey(0);
 
}
```

* 2. 均值滤波

```cpp
#include <opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
 
void MeanFilater(cv::Mat& src,cv::Mat& dst,cv::Size wsize){
	//图像边界扩充:窗口的半径
	if (wsize.height % 2 == 0 || wsize.width % 2 == 0){
		fprintf(stderr,"Please enter odd size!" );
		exit(-1);
	}
	int hh = (wsize.height - 1) / 2;
	int hw = (wsize.width - 1) / 2;
	cv::Mat Newsrc;
	cv::copyMakeBorder(src, Newsrc, hh, hh, hw, hw, cv::BORDER_REFLECT_101);//以边缘为轴，对称
	dst=cv::Mat::zeros(src.size(),src.type());
 
    //均值滤波
	int sum = 0;
	int mean = 0;
	for (int i = hh; i < src.rows + hh; ++i){
		for (int j = hw; j < src.cols + hw;++j){
 
			for (int r = i - hh; r <= i + hh; ++r){
				for (int c = j - hw; c <= j + hw;++c){
					sum = Newsrc.at<uchar>(r, c) + sum;
				}
			}
			mean = sum / (wsize.area());
			dst.at<uchar>(i-hh,j-hw)=mean;
			sum = 0;
			mean = 0;
		}
	}
 
}
 
int main(){
	cv::Mat src = cv::imread("I:\\Learning-and-Practice\\2019Change\\Image process algorithm\\Img\\Fig0334(a)(hubble-original).tif");
	if (src.empty()){
		return -1;
	}
	if (src.channels() > 1)
		cv::cvtColor(src,src,CV_RGB2GRAY);
 
	cv::Mat dst;
	cv::Mat dst1;
	cv::Size wsize(7,7);
 
	double t2 = (double)cv::getTickCount();
	MeanFilater(src, dst, wsize); //均值滤波
	t2 = (double)cv::getTickCount() - t2;
	double time2 = (t2 *1000.) / ((double)cv::getTickFrequency());
	std::cout << "FASTmy_process=" << time2 << " ms. " << std::endl << std::endl;
 
	cv::namedWindow("src");
	cv::imshow("src", src);
	cv::namedWindow("dst");
	cv::imshow("dst", dst);
	cv::imwrite("I:\\Learning-and-Practice\\2019Change\\Image process algorithm\\Image Filtering\\MeanFilter\\Mean_hubble.jpg",dst);
	cv::waitKey(0);
}
```
* 3.高斯滤波

```cpp
////////////////////////////
//按二维高斯函数实现高斯滤波
///////////////////////////
void GaussianFilter(cv::Mat& src, cv::Mat& dst, cv::Mat window){
	int hh = (window.rows - 1) / 2;
	int hw = (window.cols - 1) / 2;
	dst = cv::Mat::zeros(src.size(),src.type());
	//边界填充
	cv::Mat Newsrc;
	cv::copyMakeBorder(src, Newsrc, hh, hh, hw, hw, cv::BORDER_REPLICATE);//边界复制
	
	//高斯滤波
	for (int i = hh; i < src.rows + hh;++i){
		for (int j = hw; j < src.cols + hw; ++j){
			double sum[3] = { 0 };
 
			for (int r = -hh; r <= hh; ++r){
				for (int c = -hw; c <= hw; ++c){
					if (src.channels() == 1){
						sum[0] = sum[0] + Newsrc.at<uchar>(i + r, j + c) * window.at<double>(r + hh, c + hw);
					}
					else if (src.channels() == 3){
						cv::Vec3b rgb = Newsrc.at<cv::Vec3b>(i+r,j + c);
						sum[0] = sum[0] + rgb[0] * window.at<double>(r + hh, c + hw);//B
						sum[1] = sum[1] + rgb[1] * window.at<double>(r + hh, c + hw);//G
						sum[2] = sum[2] + rgb[2] * window.at<double>(r + hh, c + hw);//R
					}
				}
			}
 
			for (int k = 0; k < src.channels(); ++k){
				if (sum[k] < 0)
					sum[k] = 0;
				else if (sum[k]>255)
					sum[k] = 255;
			}
			if (src.channels() == 1)
			{
				dst.at<uchar>(i - hh, j - hw) = static_cast<uchar>(sum[0]);
			}
			else if (src.channels() == 3)
			{
				cv::Vec3b rgb = { static_cast<uchar>(sum[0]), static_cast<uchar>(sum[1]), static_cast<uchar>(sum[2]) };
				dst.at<cv::Vec3b>(i-hh, j-hw) = rgb;
			}
 
		}
	}
 
}
```

### 效果
<div align=center><img   width = '200' height ='200'  src ="https://img-blog.csdnimg.cn/201903282059447.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MDY0NzgxOQ==,size_16,color_FFFFFF,t_70"/></div>
<div align=center><img   width = '200' height ='200'  src ="https://img-blog.csdnimg.cn/2019032821000692.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MDY0NzgxOQ==,size_16,color_FFFFFF,t_70"/></div>   
   
   
###  相关技术文档、博客、教材、项目推荐
opencv文档: [https://docs.opencv.org/3.1.0/d4/d86/group__imgproc__filter.html#ga8c45db9afe636703801b0b2e440fce37](https://docs.opencv.org/3.1.0/d4/d86/group__imgproc__filter.html#ga8c45db9afe636703801b0b2e440fce37)    
博客：[https://blog.csdn.net/weixin_40647819/article/details/89740234](https://blog.csdn.net/weixin_40647819/article/details/89740234)  
           [https://blog.csdn.net/weixin_40647819/article/details/88774522](https://blog.csdn.net/weixin_40647819/article/details/88774522)     
python版本：[https://www.kancloud.cn/aollo/aolloopencv/269599](https://blog.csdn.net/weixin_40647819/article/details/88774522)                    [http://www.woshicver.com/FifthSection/4_4_%E5%9B%BE%E5%83%8F%E5%B9%B3%E6%BB%91/](http://www.woshicver.com/FifthSection/4_4_%E5%9B%BE%E5%83%8F%E5%B9%B3%E6%BB%91/)  

  
## 4.6 总结 

该部分对三种滤波方法进行了介绍，读者可根据提供的资料对滤波原理进行学习，然后参考示例代码自行实现。图像滤波有很多优化方法，可以提高效率，读者可以尝试学习并实现。
  
---
**Task04 图像滤波 END.**

--- ***By: 小武***
>博客：[https://blog.csdn.net/weixin_40647819](https://blog.csdn.net/weixin_40647819)


**关于Datawhale**：

>Datawhale是一个专注于数据科学与AI领域的开源组织，汇集了众多领域院校和知名企业的优秀学习者，聚合了一群有开源精神和探索精神的团队成员。Datawhale以“for the learner，和学习者一起成长”为愿景，鼓励真实地展现自我、开放包容、互信互助、敢于试错和勇于担当。同时Datawhale 用开源的理念去探索开源内容、开源学习和开源方案，赋能人才培养，助力人才成长，建立起人与人，人与知识，人与企业和人与未来的联结。
