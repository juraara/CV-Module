# Computer vision (CV) module
This repository is intended for the computer vision module for a drowsiness detection system. The drowsiness detection system is planned to be embedded in a motorcycle helmet.

## 1. Eye detection
[Haar-cascades](https://medium.com/analytics-vidhya/haar-cascades-explained-38210e57970d) will be used for the eye detection. The goal here to detect the left-most eye and determine its 2D rectangle coordinates which will be stored in a ```Rect``` variable.
```c++
    _GTON = GTON();
    CascadeClassifier eyeCascade;
    eyeCascade.load("../blinkdetection/haarcascade_eye_tree_eyeglasses.xml");
    Rect eye = _GTON.detectEyes(frame, eyeCascade, x_angle);
```
## 2. Iris detection
To explain, this simple iris detection method is based on the assumption that when the eyes are open, the largest contour present in the detected eye is the iris. This is on the assumption that the iris will always be darker than the rest of the eyes. This also holds an assumption that a normally opened eye has the top-most part of the iris touching the upper eyelid while the bottom-most part of the iris is touching the lower eyelid. And to prevent the eyelashes from ever being detected, the 2D rectangle of the detected iris will be utilized to crop out areas where the eyelashes are usually visible during eyelid opening and closure.
```c++
   _GTON = GTON();
    CascadeClassifier eyeCascade;
    eyeCascade.load("../blinkdetection/haarcascade_eye_tree_eyeglasses.xml");
    Rect eye = _GTON.detectEyes(frame, eyeCascade, x_angle);
    Rect iris = _GTON.detectIris(frame, eye);
```
A sample image is shown below of how the frame is processed starting from eye detection to iris detection.

![sample](/etc/sample.png)

## 3. Blink detection
Here, 4 light-weight blink detection algorithms will be compared; 2 of which are from unpublished papers and the other 2 are from published papers.

Note that for methods `GTON`, `CEAA` and `QSAA`, all of them assumes that when the eyes are opened, it should have a greater number of black pixels compared to when the eyes are closed. The number of black pixels is greatly affected by the presence and the absence of the iris when the eyes are opened and closed, respectively.

### a. GTON (Global Thresholding of The Negative)
This method detects eyelid closure and opening by getting the non-absolute percentage difference of the number of white pixels from the current and previous frame. A sample usage of `GTON` is shown below.

#### Step 1. Detect the eyes and the iris:
```c++
    _GTON = GTON();
    
    CascadeClassifier eyeCascade;
    eyeCascade.load("../blinkdetection/haarcascade_eye_tree_eyeglasses.xml");
    
    VideoCapture cap(0);
    
    Mat frame;
    
    while (eye.empty() || iris.empty())
    {
        cap.read(frame);
        if (frame.empty()) break;
        eye = _GTON.detectEyes(frame, eyeCascade, "0x");
        iris = _GTON.detectIris(frame, eye);
    }
```
#### Step 2. Start detecting blinks:
```c++
    while (1)
    {
        cap.read(frame);
        if (frame.empty()) break;
        frame = frame(eye);
        _GTON.detectBlink(frame, eye, iris);
    }
```
Can be initialized witht either of the following:
```c++
    _GTON = GTON();
    _GTON = GTON(double _perclos_x, double _eye_opening_perdiff, double _eye_closure_perdiff)
```

### b. CEAA (Contour Extraction and Analysis)
CEAA uses contour detection after contrasting the image to high value to the point where the iris is the only one left to produce contours. Once the iris is located, eyelid closure/opening will be based on the presence of contours from the iris.

Can be initialized with either of the following:
```c++
    _CEAA = CEAA();
    _CEAA = CEAA(double _minthresh);
```

### c. QSAA (Quick Sort and Analysis)
[QSAA](https://ieeexplore.ieee.org/abstract/document/7545182/) uses statistical analysis to determine the number of eyelid closure and openings within a period of time. To do that, the number of white pixels per frame are stored in an array or vector for a given amount of time or loading time. Said array or vector is a sliding window that moves 1 frame at a time. 

Can be initialized with either of the following:
```c++
    _QSAA = QSAA();
    _QSAA = QSAA(vector<float> _wpixels, int _load_time_f);
```

### d. EBRA (Eye Black Pixel Ratio Analysis)
[EBRA](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6174048/) splits the frame of the detected eye into 2 parts horizontally. This creates one frame containing the upper portion and another frame with the lower portion of the eye. The difference of the number of white pixels produced by the upper and lower portion of the eyes will be the basis for detecting eyelid closure and opening.

Assumes that the frame containing the upper portion of the eye should have a greater number of black pixels compared to the frame containing the lower portion of the eye, when the eye is opened. This is due to the iris and the eyelashes being present in the upper portion of the eye when the eyes are opened. When the eyes are closed, there should be a greater number of black pixels in the frame containing the lower portion of the eye. This is due to the presence of the eyelashes on the lower portion of the eye when the eyes are closed.

Can be initialized using:
```c++
    _EBRA = EBRA();
```
