#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
 using namespace std;
 using namespace cv;

 int main()
 {
	 //1.灰度模式读取图像
	 Mat srcImage=imread("D:\\ImageSource\\OW\\ow_05.jpg",0);
	 if (!srcImage.data)
	 {
		 cout<<"Error"<<endl;
		 return false;
	 }
	 imshow("Origin Image",srcImage);


	 //2.输入图像扩展
	 int m=getOptimalDFTSize(srcImage.rows);
	 int n=getOptimalDFTSize(srcImage.cols);
	 //添加像素初始化为0
	 Mat padded;
	 copyMakeBorder(srcImage,padded,0,m-srcImage.rows,0,n-srcImage.cols,BORDER_CONSTANT,Scalar::all(0));


	 //3.为傅里叶变换结果分配空间
	 //将planes数组组合成多通道数组complexI
	 Mat planes[]={Mat_<float>(padded),Mat::zeros(padded.size(),CV_32F)};
	 Mat complexI;
	 merge(planes,2,complexI);


	 //4.进行傅里叶变换
	 dft(complexI,complexI);

	 //5.复数转换为幅值
	 split(complexI,planes);
	 magnitude(planes[0],planes[1],planes[0]);
	 Mat magnitudeImage=planes[0];

	 //6.对数尺度缩放
	 magnitudeImage+=Scalar::all(1);
	 log(magnitudeImage,magnitudeImage);

	 //7.剪切和重分布幅度图象限
	 //若有奇数行和奇数列，进行频谱裁剪
	 magnitudeImage=magnitudeImage(Rect(0,0,magnitudeImage.cols & -2,magnitudeImage.rows & -2));
	 //重新排列傅里叶图像中的象限，使得原点位于图像中心
	 int cx=magnitudeImage.cols/2;
	 int cy=magnitudeImage.rows/2;
	 Mat q0(magnitudeImage,Rect(0,0,cx,cy));
	 Mat q1(magnitudeImage,Rect(cx,0,cx,cy));
	 Mat q2(magnitudeImage,Rect(0,cy,cx,cy));
	 Mat q3(magnitudeImage,Rect(cx,cy,cx,cy));
	 //交换象限左上与右下
	 Mat tmp;
	 q0.copyTo(tmp);
	 q3.copyTo(q0);
	 tmp.copyTo(q3);
	 //交换象限右上与左下
	 q1.copyTo(tmp);
	 q2.copyTo(q1);
	 tmp.copyTo(q2);


	 //8.归一化
	 normalize(magnitudeImage,magnitudeImage,0,1,CV_MINMAX);

	 //9.显示
	 imshow("频谱幅值",magnitudeImage);
	 waitKey();

	 return 0;







 }