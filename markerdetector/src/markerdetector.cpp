#include "markerdetector.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace std;

float perimeter(const std::vector<cv::Point2f>& a)
{
    float sum=0, dx, dy;

    for (size_t i=0;i<a.size();i++)
    {
        size_t i2=(i+1) % a.size();

        dx = a[i].x - a[i2].x;
        dy = a[i].y - a[i2].y;

        sum += sqrt(dx*dx + dy*dy);
    }

    return sum;
}

vector<cv::Point2f> projectPoints(const vector<cv::Point3f>& objectPoints,
                                  const cv::Mat& rvec, const cv::Mat& tvec,
                                  const cv::Mat& cameraMatrix,
                                  const cv::Mat& distCoeffs)
{
    vector<cv::Point2f> points;
    cv::projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, points);
    return points;
}

MarksDetector::MarksDetector()
    : m_markerSize{240, 240}
{
    m_markerCorners2d.push_back(cv::Point2f{0.0f,0.0f});
    m_markerCorners2d.push_back(cv::Point2f{static_cast<float>(m_markerSize.width),0.0f});
    m_markerCorners2d.push_back(cv::Point2f{static_cast<float>(m_markerSize.width),static_cast<float>(m_markerSize.height)});
    m_markerCorners2d.push_back(cv::Point2f{0.0f, static_cast<float>(m_markerSize.height)});

    cv::FileStorage fs("cameraCalibration.xml", cv::FileStorage::READ);

    fs["CameraMatrix"] >> m_cameraMatrix;
    fs["DistortionCoefficients"] >> m_distortion;

    if (!m_cameraMatrix.data || !m_distortion.data)
    {
        cerr << "Calibrate first!" << endl;
        throw std::runtime_error{"cameraCalibration file not found be sure to calibrate first"};
    }
}

void MarksDetector::processFame(cv::Mat& grayscale)
{
    m_contours.clear();
    m_possibleContours.clear();
    m_markers.clear();

    m_minCountournSize = grayscale.cols / 5;

    binarize(grayscale);
    findContours();
    findCandidates();
    recognizeCandidates();
    estimatePose();
}

const std::vector<Marker>& MarksDetector::markers() const noexcept
{
    return m_markers;
}

void MarksDetector::binarize(const cv::Mat& grayscale)
{
    m_grayscale = grayscale;
    threshold(m_grayscale, m_binarized, 127, 255.0, cv::THRESH_BINARY);
}

void MarksDetector::findContours()
{
    cv::findContours(m_binarized, m_contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
}

void MarksDetector::findCandidates()
{
    vector<cv::Point2f> approxCurve;
    vector<vector<cv::Point2f>> possibleMarkerPoints;

    // For each contour, analyze if it is a parallelepiped likely to be the marker
    for (size_t i=0; i < m_contours.size(); i++)
    {
        // Approximate to a polygon
        double eps = m_contours[i].size() * 0.05;
        cv::approxPolyDP(m_contours[i], approxCurve, eps, true);

        // We interested only in polygons that contains only four points
        if (approxCurve.size() != 4)
            continue;

        // And they have to be convex
        if (!isContourConvex(approxCurve))
            continue;

        // Ensure that the distance between consecutive points is large enough
        float minDist = std::numeric_limits<float>::max();

        for (int i = 0; i < 4; i++)
        {
            auto side = approxCurve[i] - approxCurve[(i+1)%4];
            float squaredSideLength = static_cast<float>(side.dot(side));
            minDist = std::min(minDist, squaredSideLength);
        }

        // Check that distance is not very small
        if (minDist < 500)
            continue;

        // All tests are passed. Save marker candidate:
        auto markerPoints = approxCurve;

        // Sort the points in anti-clockwise order
        // Trace a line between the first and second point.
        // If the third point is at the right side, then the points are anti-clockwise
        auto v1 = markerPoints[1] - markerPoints[0];
        auto v2 = markerPoints[2] - markerPoints[0];

        double o = (v1.x * v2.y) - (v1.y * v2.x);

        if (o < 0.0)		 // if the third point is in the left side, then sort in anti-clockwise order
            std::swap(markerPoints[1], markerPoints[3]);

        possibleMarkerPoints.emplace_back(markerPoints);
    }

    // calculate the average distance of each corner to the nearest corner of the other marker candidate
    std::vector< std::pair<int,int> > tooNearCandidates;
    for (int i=0;i<possibleMarkerPoints.size();i++)
    {
        const auto& points1 = possibleMarkerPoints[i];

        //calculate the average distance of each corner to the nearest corner of the other marker candidate
        for (int j=i+1;j<possibleMarkerPoints.size();j++)
        {
            const auto& points2 = possibleMarkerPoints[j];

            float distSquared = 0;

            for (int c = 0; c < 4; c++)
            {
                auto v = points1[c] - points2[c];
                distSquared += v.dot(v);
            }

            distSquared /= 4;

            if (distSquared < 100)
                tooNearCandidates.push_back(std::pair<int,int>(i,j));
        }
    }

    // Mark for removal the element of the pair with smaller perimeter
    std::vector<bool> removalMask (possibleMarkerPoints.size(), false);

    for (size_t i = 0; i < tooNearCandidates.size(); i++)
    {
        float p1 = perimeter(possibleMarkerPoints[tooNearCandidates[i].first ]);
        float p2 = perimeter(possibleMarkerPoints[tooNearCandidates[i].second]);

        size_t removalIndex;
        if (p1 > p2)
            removalIndex = tooNearCandidates[i].second;
        else
            removalIndex = tooNearCandidates[i].first;

        removalMask[removalIndex] = true;
    }

    for (size_t i = 0; i < possibleMarkerPoints.size(); i++)
        if (!removalMask[i])
            m_possibleContours.push_back(possibleMarkerPoints[i]);
}

void MarksDetector::recognizeCandidates()
{
    for(auto& points: m_possibleContours)
    {
        cv::Mat canonicalMarkerImage;

        // Find the perspective transformation that brings current marker to rectangular form
        cv::Mat markerTransform = getPerspectiveTransform(points, m_markerCorners2d);

        // Transform image to get a canonical marker image
        cv::warpPerspective(m_binarized, canonicalMarkerImage,  markerTransform, m_markerSize);

        Marker m(canonicalMarkerImage, points);

        if (!m.isValid())
            continue;

        auto termCriteria = cv::TermCriteria{cv::TermCriteria::MAX_ITER |
                            cv::TermCriteria::EPS, 30, 0.01};
        cornerSubPix(m_grayscale, points, cv::Size{5, 5}, cv::Size{-1, -1}, termCriteria);

        m.precisePoints(points);
        m_markers.push_back(m);
    }
}

void MarksDetector::estimatePose()
{
    for(Marker& m : m_markers)
    {
        vector<cv::Point3f> objectPoints = {
            cv::Point3f{-1, -1, 0},
            cv::Point3f{-1, 1, 0},
            cv::Point3f{1, 1, 0},
            cv::Point3f{1, -1, 0}
        };

        cv::Mat objectPointsMat(objectPoints);
        cv::Mat rvec, tvec;
        solvePnP(objectPointsMat, m.points(), m_cameraMatrix, m_distortion, rvec, tvec);

        vector<vector<cv::Point3f>> lineIn3D =
        {
            {{-1.0f, -1.0f, 0.0f}, {-1.0f, -1.0f, 2.0f}},
            {{-1.0f,  1.0f, 0.0f}, {-1.0f,  1.0f, 2.0f}},
            {{ 1.0f, -1.0f, 0.0f}, { 1.0f, -1.0f, 2.0f}},
            {{ 1.0f,  1.0f, 0.0f}, { 1.0f,  1.0f, 2.0f}},
            {{-1.0f,  1.0f, 2.0f}, { 1.0f,  1.0f, 2.0f}},
            {{-1.0f, -1.0f, 2.0f}, { 1.0f, -1.0f, 2.0f}},
            {{-1.0f,  1.0f, 2.0f}, {-1.0f, -1.0f, 2.0f}},
            {{ 1.0f,  1.0f, 2.0f}, { 1.0f, -1.0f, 2.0f}}
        };

        vector<vector<cv::Point2f>> lineIn2D;
        lineIn2D.reserve(lineIn3D.size());

        std::transform(begin(lineIn3D), end(lineIn3D), back_inserter(lineIn2D),
                       [&,this](const vector<cv::Point3f>& points3D)
        {
            return projectPoints(points3D, rvec, tvec, m_cameraMatrix, m_distortion);
        });

        m.setCube(lineIn2D);
    }
}
