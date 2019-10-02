# Writeup - P2 Advaced Lane Line Finding
---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/corner16.jpg "Corner"
[image2]: ./output_images/original7.jpg "Original Cal Image"
[image3]: ./output_images/undistort_test7.jpg "Undistort Cal Image"
[image4]: ./output_images/wraped1.jpg "Warp Example"
[image5]: ./output_images/masked1.jpg "Masked Image"
[image6]: ./output_images/window.jpg "Sliding Windows"
[image7]: ./output_images/final1.jpg "Output"
[image8]: ./output_images/original6.jpg "Original Cal Image"
[image9]: ./output_images/undistort_test6.jpg "Undistort Cal Image"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./P2_Advaced_Lane_Finding.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  
First I've started with nx = 9 and ny = 6 corners, but after running the calibration for the first time I've noticed that some test-images doesn't show the full chessboard with all the corners. Due to this situation I've integrated a check for fewer corners (`9 & 5` or `8 & 6` and so on) in case the algorithm can't detect the full amount of corners.

Detected corner points are shown in the picture below:
![Detected corner points][image1]


Afterwards I've used the resulting `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

#### Original Image
![Original Image][image2]

#### Undistort Image
![Original Image][image3]


The calibration results were saved as a `pickle` for further usage in the future.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Distortion matrix was also applied on the test images from the road. The differences are difficult to see, but are slightly visible in the corner area. Examples are shown below:

#### Original Image
![Original Image][image8]

#### Undistort Image
![Original Image][image9]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. Thresholding steps are described at lines #22 through #53 of the `thresholding()` function in `"./P2_Advaced_Lane_Finding.ipynb`.  
In order to create a good mask for the original image and clearly identify the lane lines I've used the combination of color and gradient threshold. As gradient threshold I've used the `sobel_x` (Sobel gradient in X direction) since I've noticed that the visibility of the lines is better with sobel_x gradient.
For color threshold I've selected the yellow and the white lanes using different colorspaces (`RGB, HSV, HLS`). From the `HLS` channel I've also applied a threshold on the "S" channel in order to separate the lane lines.
Threshold for each filter was checked and adjusted till the results were acceptable for me.
Finally I've binary combined all the masks to the final, combined mask.  

Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![Masked Image][image5]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.
Perspective transform ("birds-eye view") code is listed under chapter `4` in the `.P2_Advaced_Lane_Finding.ipynb` notebook.
In the first step I've took the test image with the straight lines in order to define the source points (it's easier to define them on the image with straight lines). I've read out the coordinates manually and noted them in the code line `9`.
For the destination points `dst` I've selected the coordinates in order to create a fictive rectangle for the lane lines in the current image.

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 200, 720     | 250, 720        |
| 1100, 720      | 950, 720      |
| 715, 465     | 250, 0      |
| 565, 465      | 950, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

The source and destination points are used as input for the `PerspectiveTransform()` function which calculates and returns the transform matrix and the iverse transform matrix. This output is required for the opencv function `cv2.warpPerspective()` in order to calculate the warped ("birds-eye view") image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In order to identify the lines I've first calculated the histogram of the bottom half of the image. With the help of the histogram I was able to identify the peaks which are linked to the left and right lane lines. These are the starting points `leftx_base` or `rightx_base` for the lines.
I've integrated the function `sliding_window()`, where the image is divided in the 'nwindows'. In these windows we look (left & right) for the pixel position and recenter the windows if required. Example is shown in the image below:

![alt text][image6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Radius of the curvature was calculated based on the formula provided in the exercise. I've applied the formula in my script as part of the function ``curvature()``, in lines `#22 - 23`.

``` python
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
```
Result of the calculated was converted from pixel in meter.

As soon as the left and right lanes are defined, it's easy to calculate the center line (middle of left & right). Calculation of the off-center position of the vehicle is based on the assumption, that the camera is mounted in the middle of the car. In this case you can calculate the offset of the image-center and the centerline.


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I've implemented this step in the `line_plot()` function of my algorithm. Results are shown in the image below:

![alt text][image7]

---
### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One of the issues I had was during the detection of the corner during the camera calibration. I recognized that in some pictures the cornerpoint were not detected. That I've noticed that the provided images partially doesn't have the full corner amount. So I've implemented an algorithm which checks for fewer corner number `nx`and `ny` in case the original amount of corners was not detected.
I've also struggled during the definition of the threshold for each mask. I think a hard coded variant as it's implemented in the moment is not the best, since I expect that the mask could have some issues in different light conditions.
I've also noticed that sometime the polyfit() function doesn't deliver optimal results.
In order to make the algorithm more robust I would propose to put more time in the mask & threshold definition, in order to define the perfect parameters. Machine learning technics can be helpful for this purpose.
