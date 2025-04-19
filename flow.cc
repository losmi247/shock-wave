#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char *argv[]) {
    if(argc != 4) {
        cerr << "Incorrect usage: ./flow <video file> <frame number> <output file name>" << endl;
        return -1;
    }

    // Load the video
    VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
        cerr << "Error opening video file." << endl;
        return -1;
    }

    // Seek to frame
    cap.set(CAP_PROP_POS_FRAMES, stoi(argv[2]));

    Mat frame1_color, frame2_color, frame1, frame2;
    cap >> frame1_color;
    cap >> frame2_color;

    if (frame1_color.empty() || frame2_color.empty()) {
        cerr << "Could not load frames." << endl;
        return -1;
    }

    // Convert to grayscale
    cvtColor(frame1_color, frame1, COLOR_BGR2GRAY);
    cvtColor(frame2_color, frame2, COLOR_BGR2GRAY);

    // Compute optical flow using Farneback
    Mat flow;
    calcOpticalFlowFarneback(frame1, frame2, flow,
                             0.5, 3, 15, 3, 5, 1.2, 0);

    // Visualize flow as arrows
    Mat vis = frame1_color.clone();
    for (int y = 0; y < vis.rows; y += 20) {
        for (int x = 0; x < vis.cols; x += 20) {
            const Point2f& fxy = flow.at<Point2f>(y, x);
    
            // Compute magnitude and angle
            float mag = sqrt(fxy.x * fxy.x + fxy.y * fxy.y);
            float angle = atan2(fxy.y, fxy.x) * 180 / CV_PI;
    
            // Set color based on angle (optional: map to HSV for variety)
            Scalar color = Scalar(255, 0, 255); // static magenta, or map from angle
    
            // Scale the vector for visual trail
            float scale = 10.0;
            Point start(x, y);
            Point end(cvRound(x + scale * fxy.x), cvRound(y + scale * fxy.y));
    
            line(vis, start, end, color, 1, LINE_AA);
            circle(vis, start, 2, color, -1);
        }
    }    

    char output_pathname[100];
    snprintf(output_pathname, 100, "./results/%s.png", argv[3]);
    imwrite(output_pathname, vis);
    return 0;
}