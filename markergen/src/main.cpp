#include "options.hpp"
#include <boost/crc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

Mat addFrame(cv::Mat& image, const cv::Size& blockSize);
Mat addOrientation(cv::Mat& image, const Size &blockSize);
void encodeLine(cv::Mat& dataImageLine, const cv::Size& blockSize, int bitCount, uint64_t bits);
void encode(cv::Mat& image, uint64_t id, const Size &blockSize);

int main(int argc, char* argv[])
{
    Options opt{argc, argv};

    if (opt.showOnlyHelp())
    {
        cout << opt << endl;
        return EXIT_SUCCESS;
    }

    constexpr auto BLOCK_COUNT = 12;
    Size blockSize{opt.blockWidth(), opt.blockWidth()};
    Size imageSize = blockSize * BLOCK_COUNT;

    Mat image{imageSize, CV_8UC1, Scalar{0.0}};

    auto border = addFrame(image, blockSize);
    auto data = addOrientation(border, blockSize);
    encode(data, opt.code(), blockSize);

    cv::imshow("Marker", image);
    cv::waitKey(0);

    imwrite(opt.filename(), image);
}

cv::Mat addFrame(cv::Mat& image, const cv::Size& blockSize)
{
    Mat vline{Size{image.cols, blockSize.height}, image.type(), cv::Scalar{0}};
    vline.copyTo(image(Rect{0,0, image.cols, blockSize.height}));
    vline.copyTo(image(Rect{0, image.rows - blockSize.height, image.cols, blockSize.height}));

    Mat hline{Size{blockSize.width, image.rows}, image.type(), cv::Scalar{0}};
    hline.copyTo(image(Rect{0,0, blockSize.width, image.cols}));
    hline.copyTo(image(Rect{image.rows - blockSize.width, 0, blockSize.width, image.cols}));

    auto orientationImage = image(Rect{
                                      blockSize.width, blockSize.height,
                                      image.cols - 2*blockSize.width, image.rows - 2*blockSize.height
                                  });

    return orientationImage;
}

cv::Mat addOrientation(cv::Mat& image, const cv::Size& blockSize)
{
    Mat whiteSquare{blockSize, image.type(), cv::Scalar{255.0, 255.0, 255.0}};
    whiteSquare.copyTo(image(Rect{0,0, blockSize.width, blockSize.height}));
    whiteSquare.copyTo(image(Rect{image.cols - blockSize.width,0, blockSize.width, blockSize.height}));
    whiteSquare.copyTo(image(Rect{0, image.rows - blockSize.height, blockSize.width, blockSize.height}));

    auto dataImage = image(
                Rect{
                    blockSize.width, blockSize.height,
                    image.cols - 2*blockSize.width, image.rows - 2*blockSize.height
                }
                );

    return dataImage;
}

void encode(cv::Mat& image, uint64_t id, const cv::Size& blockSize)
{
    const auto bitmask = 0xff;
    const auto bitCount = 8;

    for(int i = 0; i < 6; ++i)
    {
        auto dataImageLine = image(
                    Rect{
                        0, i * blockSize.height,
                        image.cols, blockSize.height
                    }
                    );

        auto bits = (id & (bitmask << i*bitCount)) >> i*bitCount;
        encodeLine(dataImageLine, blockSize, bitCount, bits);
    }

    boost::crc_16_type crc;
    crc.process_bytes(&id, sizeof(id));

    auto checksum = crc.checksum();

    auto crc1 = checksum & 0xFF;
    auto crc2 = checksum >> 8;

    cout << hex << "checksum is: " << checksum << "( " << crc1 << " - " << crc2 << " )" << dec << endl;

    auto crc1LineImage = image(
                Rect{
                    0, 6 * blockSize.height,
                    image.cols, blockSize.height
                }
                );

    auto crc2LineImage = image(
                Rect{
                    0, 7 * blockSize.height,
                    image.cols, blockSize.height
                }
                );

    encodeLine(crc1LineImage, blockSize, bitCount, crc1);
    encodeLine(crc2LineImage, blockSize, bitCount, crc2);
}

void encodeLine(cv::Mat& dataImageLine, const cv::Size& blockSize, int bitCount, uint64_t bits)
{
    for(int j = 0; j < bitCount; ++j)
    {
        uint64_t mask = uint64_t(1) << j;
        uint64_t bit = bits & mask;

        auto color = bit == 0 ? Scalar{0} : Scalar{255, 255, 255};
        cv::Mat square{blockSize, dataImageLine.type(), color};

        square.copyTo(
                    dataImageLine(
                        Rect{
                            j * blockSize.width, 0,
                            blockSize.width, blockSize.height}
                        )
                    );
    }
}
