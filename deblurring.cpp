#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;

#define MAX_ITERS_KERNEL_SIZE 10
#define THRESH_PARAM 3.5                //threshold value(kernel size estimation) = maxValue / THRESH_PARAM
#define FILTER_SIZE_PARAM 10            //filter size = imageSize / FILTER_SIZE_PARAM
#define BLUR_STD_DEV 20                 //image blurred if stddev(laplacian) < BLUR_STD_DEV

void computeDFT(const Mat &image, Mat *result);
void computeIDFT(Mat *input, Mat &result);
void wienerFilter(const Mat &blurredImage, const Mat &known, Mat &unknown, float k);
void rotate(const Mat &src, Mat &dst);
void blindDeblurringOneChannel(const Mat &blurred, Mat &deblurred, Mat &kernel, int kernelSize, int iters, float noisePower);
void blindDeblurring(const Mat &blurred, Mat &deblurred, Mat &kernel, int iters, float noisePower);
void applyConstraints(Mat &image);
Mat getAutoCerrelation(const Mat &blurred);
int estimateKernelSize(const Mat &blurred);
void cropBorder(Mat &image);
float measureBlur(const Mat &grayBlurred);
bool isBlurred(const Mat &grayBlurred);

int main(int argc, char* argv[])
{
    String file;
    file = argv[1];
    Mat blurred = imread(file, CV_LOAD_IMAGE_COLOR);
    Mat deblurred;
    Mat kernel;
    blindDeblurring(blurred, deblurred, kernel, 100, 0.02);
    imwrite(argv[2], deblurred);
    return 0;
}

float measureBlur(const Mat &grayBlurred)
{
    Mat dst;
    Laplacian(grayBlurred, dst, -1, 3, 1, 0, BORDER_CONSTANT);
    dst.convertTo(dst, CV_8UC1);
    float sum = 0;
    for (int i = 0; i < grayBlurred.rows; i++)
    {
        for (int j = 0; j < grayBlurred.cols; j++)
        {
            sum+=dst.at<unsigned char>(i, j);
        }
    }
    sum/=(grayBlurred.rows * grayBlurred.cols);
    cout << sum << endl;
    return sum;
}

bool isBlurred(const Mat &grayBlurred)
{
    return measureBlur(grayBlurred) < BLUR_STD_DEV;
}

Mat getAutoCerrelation(const Mat &blurred)
{
    Mat dst;
    int ddepth = CV_16S;
    Laplacian(blurred, dst, -1, 3, 1, 0, BORDER_CONSTANT);
    dst.convertTo(dst, CV_32FC1);
    Mat correlation;
    filter2D(dst, correlation, -1, dst, Point(-1,-1), 0, BORDER_CONSTANT);
    return correlation.clone();
}

void cropBorder(Mat &thresholded)
{
    int top = 0;
    int left = 0;
    int right = 0;
    int bottom = 0;
    for (int i = 0; i < thresholded.rows; i++)
    {
        for (int j = 0; j < thresholded.cols; j++)
        {
            if (thresholded.at<unsigned char>(i ,j) == 255)
            {
                top = i;
                break;
            }
        }
        if (top)
            break;
    }

    for (int i = thresholded.rows - 1; i > 0; i--)
    {
        for (int j = 0; j < thresholded.cols; j++)
        {
            if (thresholded.at<unsigned char>(i ,j) == 255)
            {
                bottom = i;
                break;
            }
        }
        if (bottom)
            break;
    }

    for (int i = 0; i < thresholded.cols; i++)
    {
        for (int j = 0; j < thresholded.rows; j++)
        {
            if (thresholded.at<unsigned char>(j ,i) == 255)
            {
                left = i;
                break;
            }
        }
        if (left)
            break;
    }

    for (int i = thresholded.cols - 1; i > 0; i--)
    {
        for (int j = 0; j < thresholded.rows; j++)
        {
            if (thresholded.at<unsigned char>(j, i) == 255)
            {
                right = i;
                break;
            }
        }
        if (right)
            break;
    }
    thresholded = thresholded(Rect(left, top, right - left, bottom - top));
}

int estimateKernelSize(const Mat &grayBlurred)
{
    int kernelSize = 0;
    Mat correlation = getAutoCerrelation(grayBlurred);
    Mat thresholded = correlation.clone();
    for(int i = 0; i < MAX_ITERS_KERNEL_SIZE; i++)
    {
        Mat thresholdedNEW;
        double minVal;
        double maxVal;
        minMaxLoc(thresholded, &minVal, &maxVal);
        threshold(thresholded, thresholded, round(maxVal / THRESH_PARAM), 255, THRESH_BINARY);
        thresholded.convertTo(thresholded, CV_8UC1);
        cropBorder(thresholded);
        if (thresholded.rows < 3)
        {
            break;
        }
        int filterSize = (int)(max(3, ((thresholded.rows + thresholded.cols) / 2)/FILTER_SIZE_PARAM));
        if (!(filterSize & 1))
        {
            filterSize++;
        }
        Mat filter = Mat::ones(filterSize, filterSize, CV_32FC1) / (float)(filterSize * filterSize - 1);
        filter.at<float>(filterSize / 2, filterSize / 2) = 0;
        filter2D(thresholded, thresholdedNEW, -1, filter, Point(-1, -1), 0, BORDER_CONSTANT);
        kernelSize = (thresholdedNEW.rows + thresholdedNEW.cols) / 2;
        if (!(kernelSize & 1))
        {
            kernelSize++;
        }
        thresholded = thresholdedNEW.clone();
    }
    cout << "estimated kernel size " << kernelSize << endl;
    return kernelSize;
}

void computeDFT(const Mat &image, Mat *result)
{
    Mat padded;
    int m = getOptimalDFTSize(image.rows);
    int n = getOptimalDFTSize(image.cols);
    copyMakeBorder(image, padded, 0, m - image.rows, 0, n - image.cols, BORDER_CONSTANT, Scalar::all(0));
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32FC1)};
    Mat fimg;
    merge(planes, 2, fimg);
    dft(fimg, fimg);
    split(fimg, planes);
    planes[0] = planes[0](Rect(0, 0, image.cols, image.rows));
    planes[1] = planes[1](Rect(0, 0, image.cols, image.rows));
    result[0] = planes[0].clone();
    result[1] = planes[1].clone();
}

void computeIDFT(Mat *input, Mat &result)
{
    Mat fimg;
    merge(input, 2, fimg);
    Mat inverse;
    idft(fimg, inverse, DFT_REAL_OUTPUT + DFT_SCALE);
    result = inverse.clone();
}

void rotate(Mat &src, Mat &dst)
{
    int cx = src.cols >> 1;
    int cy = src.rows >> 1;
    Mat tmp;
    tmp.create(src.size(), src.type());
    src(Rect(0, 0, cx, cy)).copyTo(tmp(Rect(cx, cy, cx, cy)));
    src(Rect(cx, cy, cx, cy)).copyTo(tmp(Rect(0, 0, cx, cy)));
    src(Rect(cx, 0, cx, cy)).copyTo(tmp(Rect(0, cy, cx, cy)));
    src(Rect(0, cy, cx, cy)).copyTo(tmp(Rect(cx, 0, cx, cy)));
    dst = tmp.clone();
}

void applyConstraints(Mat &image)
{
    for (int i = 0; i < image.rows; i++){
        for (int j = 0; j < image.cols; j++){
            if (image.at<float>(i, j) < 0)
            {
                image.at<float>(i, j) = 0;
            }
            if (image.at<float>(i, j) > 255)
            {
                image.at<float>(i, j) = 255.0;
            }
        }
    }
}

void normalizePSF(Mat &image)
{
    float sum = 0;
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            sum+=image.at<float>(i, j);
        }
    }
    image/=sum;
}

void blindDeblurring(const Mat &blurred, Mat &deblurred, Mat &kernel, int iters, float noisePower)
{
    Mat grayBlurred;
    cvtColor(blurred, grayBlurred, CV_BGR2GRAY);
    if (!isBlurred(grayBlurred))
    {
        cout << "not blurred" << endl;
        deblurred = blurred.clone();
        kernel = Mat::zeros(3, 3, CV_8UC1);
        kernel.at<unsigned char>(1, 1) = 1;
        return;
    }
    Mat grayDeblurred;
    vector<Mat> blurredRGB(3);
    split(blurred, blurredRGB);
    vector<Mat> deblurredRGB(3);
    int kernelSize = estimateKernelSize(grayBlurred);
    Mat resultDeblurred;
    Mat resultKernel;
    blindDeblurringOneChannel(grayBlurred, grayDeblurred, resultKernel, kernelSize, iters, noisePower);
    for (int i = 0; i < 3; i++)
    {
        wienerFilter(blurredRGB[i], resultKernel, deblurredRGB[i], noisePower);
    }
    merge(deblurredRGB, resultDeblurred);
    deblurred = resultDeblurred.clone();
    kernel = resultKernel.clone();
}

void blindDeblurringOneChannel(const Mat &blurred, Mat &deblurred, Mat &kernel, int kernelSize, int iters, float noisePower)
{
    Mat kernelCurrent = Mat::ones(kernelSize, kernelSize, CV_32FC1);
    kernelCurrent/=(kernelSize * kernelSize);
    Mat deblurredCurrent = blurred.clone();
    for (int i = 0; i < iters; i++)
    {
        wienerFilter(blurred, kernelCurrent.clone(), deblurredCurrent, noisePower);
        wienerFilter(blurred, deblurredCurrent.clone(), kernelCurrent, noisePower);
        kernelCurrent = kernelCurrent(Rect((blurred.cols - kernelSize)/2 ,(blurred.rows - kernelSize)/2, kernelSize, kernelSize));
        normalizePSF(kernelCurrent);
    }
    deblurred = deblurredCurrent.clone();
    kernel = kernelCurrent.clone();
}

void wienerFilter(const Mat &blurredImage, const Mat &known, Mat &unknown, float noisePower)
{
    int imageWidth = blurredImage.size().width;
    int imageheight = blurredImage.size().height;
    Mat yFT[2];
    computeDFT(blurredImage, yFT);

    Mat padded = Mat::zeros(imageheight, imageWidth, CV_32FC1);
    int padx = padded.cols - known.cols;
    int pady = padded.rows - known.rows;
    copyMakeBorder(known, padded, pady / 2, pady - pady / 2, padx / 2, padx - padx / 2, BORDER_CONSTANT, Scalar::all(0));
    Mat paddedFT[2];
    computeDFT(padded, paddedFT);

    Mat tempUnknown;
    Mat unknownFT[2];
    unknownFT[0] = Mat::zeros(imageheight, imageWidth, CV_32FC1);
    unknownFT[1] = Mat::zeros(imageheight, imageWidth, CV_32FC1);

    float paddedRe;
    float paddedIm;
    float paddedABS;
    float denominator;
    complex<float> numerator;

    for (int i = 0; i < padded.rows; i++)
    {
        for (int j = 0; j < padded.cols; j++)
        {
            paddedRe = paddedFT[0].at<float>(i,j);
            paddedIm = paddedFT[1].at<float>(i,j);
            paddedABS = paddedRe * paddedRe + paddedIm * paddedIm;
            denominator = noisePower + paddedABS;
            numerator = complex<float>(paddedRe, -paddedIm) * complex<float>(yFT[0].at<float>(i,j), yFT[1].at<float>(i,j));
            unknownFT[0].at<float>(i,j) = numerator.real() / denominator;
            unknownFT[1].at<float>(i,j) = numerator.imag() / denominator;
        }
    }
    computeIDFT(unknownFT, tempUnknown);
    applyConstraints(tempUnknown);
    rotate(tempUnknown, tempUnknown);
    unknown = tempUnknown.clone();
}