//
// Created by yephiu on 2023/5/13.
//

#include "my_lime.h"
#include <string>
#include <opencv2/imgproc.hpp>



using namespace cv;
using namespace std;

/**
 * 计算图像在各个方向上的梯度
 * @param t 输入的照度图
 * @param hat_t 输入的估计照度图
 * @param x X方向
 * @param y y方向
 * @param dx x方向上的微分
 * @param dy y方向上的微分
 * @return
 */
float computeGradient(const Mat &t, const Mat &hat_t, int x, int y, int dx, int dy)
{

    if (x + dx < 0 || x + dx >= t.cols || y + dy < 0 || y + dy >= t.rows)
    {
        return 0.0f;
    }
    float gradient = (t.at<float>(y + dy, x + dx) - t.at<float>(y, x)) * (hat_t.at<float>(y + dy, x + dx) - hat_t.at<float>(y, x));
    return gradient;
}

/**
 * 计算优化照度图
 * @param t 照度图
 * @param num_iterations 卷积次数
 * @param alpha 系数
 * @return
 */
Mat computeOptimizedIllumination(const Mat &t, int num_iterations = 100, float alpha = 0.01)
{
    Mat hat_t = t.clone();

    for (int iter = 0; iter < num_iterations; ++iter)
    {
        for (int y = 0; y < t.rows; ++y)
        {
            for (int x = 0; x < t.cols; ++x)
            {
                float gradient = 0.0f;
                for (int dy = -1; dy <= 1; ++dy)
                {
                    for (int dx = -1; dx <= 1; ++dx)
                    {
                        if (dx == 0 && dy == 0) continue;
                        gradient += computeGradient(t, hat_t, x, y, dx, dy);
                    }
                }

                hat_t.at<float>(y, x) += alpha * gradient;
            }
        }
    }

    return hat_t;
}

/**
 * 照度图计算
 * @param src 输入图像
 * @param filter_size 均值滤波系数
 * @return
 */
Mat Illu_Map(Mat src,int filter_size)
{
    int rows = src.size().height;
    int cols = src.size().width;

    src.convertTo(src,CV_32FC3);
    Mat temp=Mat::zeros(src.size(),CV_32FC1);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            temp.at<float>(i,j)= max(src.at<Vec3f>(i,j).val[0],max(src.at<Vec3f>(i,j).val[1],src.at<Vec3f>(i,j).val[2]));
        }
    }

    normalize(temp, temp, 0, 1, NORM_MINMAX, CV_32F);

    Mat kernel = Mat::ones(filter_size, filter_size, CV_32F) / (float)(filter_size * filter_size);
    filter2D(temp, temp, -1, kernel, Point(-1, -1), 0, BORDER_REPLICATE);
    return temp;
}

/**
 * 期望回复图像计算
 * @param L 原图
 * @param T_hat 优化后的照度图
 * @return
 */
Mat Recover(Mat L,Mat T_hat)
{
    const float epsilon = 0.01f;
    L.convertTo(L,CV_32FC3);
    int rows = L.size().height;
    int cols = L.size().width;
    Mat res=Mat::zeros(L.size(),CV_32FC3);

    for (int i = 0; i < rows;i++) {
        for (int j = 0; j < cols; j++) {
            for (int k = 0; k < 3; k++) {
                res.at<Vec3f>(i,j).val[k]=L.at<Vec3f>(i,j).val[k]/(T_hat.at<float>(i,j)+epsilon);
            }
        }
    }

    normalize(res, res, 0, 1, NORM_MINMAX, CV_32F);
    return res;
}

/**
 * 对优化后的找照度图进行gamma变换
 * @param srcImage 待变换图像
 * @param dstImage 变换后图像的储存位置
 * @param gamma gamma系数
 */
void GammaTransform(const Mat& srcImage,Mat& dstImage,double gamma)
{
    Mat inputFloatImage;
    srcImage.convertTo(inputFloatImage,CV_32F);
    cv::pow(inputFloatImage,gamma,dstImage);
}


/**
 * 实现LIME图像增强
 * @param L 输入图像
 * @return 增强好的图像
 */
Mat enhance(Mat L)
{
    double  begin,end,ret;

    int rows = L.size().height;
    int cols = L.size().width;

    Mat T_hat=Mat::zeros(L.size(),CV_32FC1);
    Mat T_refine=Mat::zeros(L.size(),CV_32FC1);
    Mat res=Mat::zeros(L.size(),CV_32FC3);

    L.convertTo(L,CV_32FC3);


    T_hat=Illu_Map(L,10);



    T_refine=computeOptimizedIllumination(T_hat);


    Mat R=Mat::zeros(L.size(),CV_32FC3);
    R=Recover(L,T_refine);


    Mat T_Gamma=Mat::zeros(L.size(),CV_32F);
    GammaTransform(T_refine,T_Gamma,0.5);



    for (int i = 0; i < rows;i++) {
        for (int j = 0; j < cols; j++) {
            for (int k = 0; k < 3; k++) {
                res.at<Vec3f>(i, j).val[k] = R.at<Vec3f>(i, j).val[k] * T_Gamma.at<float>(i, j);
            }
        }
    }

    //归一化
    normalize(res, res, 0, 1, NORM_MINMAX, CV_32F);
    res.convertTo(res,CV_32FC3,2.0,0.2);
    res.convertTo(res,CV_8UC3,255);

    return res;
}




