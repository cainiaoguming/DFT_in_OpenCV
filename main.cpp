#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
 using namespace std;
 using namespace cv;

 int main()
 {
	 //1.�Ҷ�ģʽ��ȡͼ��
	 Mat srcImage=imread("D:\\ImageSource\\OW\\ow_05.jpg",0);
	 if (!srcImage.data)
	 {
		 cout<<"Error"<<endl;
		 return false;
	 }
	 imshow("Origin Image",srcImage);


	 //2.����ͼ����չ
	 int m=getOptimalDFTSize(srcImage.rows);
	 int n=getOptimalDFTSize(srcImage.cols);
	 //������س�ʼ��Ϊ0
	 Mat padded;
	 copyMakeBorder(srcImage,padded,0,m-srcImage.rows,0,n-srcImage.cols,BORDER_CONSTANT,Scalar::all(0));


	 //3.Ϊ����Ҷ�任�������ռ�
	 //��planes������ϳɶ�ͨ������complexI
	 Mat planes[]={Mat_<float>(padded),Mat::zeros(padded.size(),CV_32F)};
	 Mat complexI;
	 merge(planes,2,complexI);


	 //4.���и���Ҷ�任
	 dft(complexI,complexI);

	 //5.����ת��Ϊ��ֵ
	 split(complexI,planes);
	 magnitude(planes[0],planes[1],planes[0]);
	 Mat magnitudeImage=planes[0];

	 //6.�����߶�����
	 magnitudeImage+=Scalar::all(1);
	 log(magnitudeImage,magnitudeImage);

	 //7.���к��طֲ�����ͼ����
	 //���������к������У�����Ƶ�ײü�
	 magnitudeImage=magnitudeImage(Rect(0,0,magnitudeImage.cols & -2,magnitudeImage.rows & -2));
	 //�������и���Ҷͼ���е����ޣ�ʹ��ԭ��λ��ͼ������
	 int cx=magnitudeImage.cols/2;
	 int cy=magnitudeImage.rows/2;
	 Mat q0(magnitudeImage,Rect(0,0,cx,cy));
	 Mat q1(magnitudeImage,Rect(cx,0,cx,cy));
	 Mat q2(magnitudeImage,Rect(0,cy,cx,cy));
	 Mat q3(magnitudeImage,Rect(cx,cy,cx,cy));
	 //������������������
	 Mat tmp;
	 q0.copyTo(tmp);
	 q3.copyTo(q0);
	 tmp.copyTo(q3);
	 //������������������
	 q1.copyTo(tmp);
	 q2.copyTo(q1);
	 tmp.copyTo(q2);


	 //8.��һ��
	 normalize(magnitudeImage,magnitudeImage,0,1,CV_MINMAX);

	 //9.��ʾ
	 imshow("Ƶ�׷�ֵ",magnitudeImage);
	 waitKey();

	 return 0;







 }