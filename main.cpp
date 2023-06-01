#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <my_lime.h>


using namespace cv;
using namespace std;


int main()
{
    Mat img_in;
    img_in = imread("../pic/cars.jpg");

    Mat img_out;

    img_out=enhance(img_in);

    imshow("src",img_in);
    imshow("enhance by LIME",img_out);
    waitKey(0);
    return 0;
}



