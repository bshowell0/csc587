### Overall Summary

This repository, `disc-metrics`, provides a comprehensive computer vision tool for analyzing disc golf throws from a standard video recording. Its primary goal is to offer a low-cost, accessible alternative to expensive equipment like radar guns or specialized hardware (e.g., TechDisc) for measuring key performance metrics. By processing a video, the program automatically calculates the disc's release speed and launch angle, which are critical factors for achieving greater throwing distance. Beyond flight metrics, it also performs detailed biomechanical analysis by tracking the thrower's body position using a machine learning model, culminating in a 3D wireframe visualization of the throwing motion.

The project's workflow is orchestrated by the main script, `process.py`, and is divided into several distinct stages:

1.  **Perspective Calibration**: To convert pixel measurements into real-world units (meters), the system first calibrates itself. It analyzes the first few frames of the video, where the user is expected to hold the disc up. Using OpenCV's Hough Circle Transform, it detects the disc and, given a known real-world radius (defaulted to 13.6525 cm), calculates a `pixels-to-meters` ratio. This is a clever approach that avoids the need for a separate calibration object in the field of view.

2.  **Disc Flight Tracking**: The system then focuses on the portion of the video where the disc is in flight. It first generates a static background image by averaging frames. Then, for each subsequent frame, it isolates the moving disc by subtracting the background, thresholding the difference, and finding the contours of the resulting object. This process yields the disc's (x, y) coordinates in each frame.

3.  **Metric Calculation**: With a time-series of the disc's pixel positions, the system calculates the pixel distance traveled between frames. Using the previously determined calibration ratio and the video's frames-per-second (FPS), it converts these pixel velocities into real-world speed (m/s and mph). It also calculates the average launch angle relative to the horizontal. Outlier data points are filtered to ensure a robust and accurate final measurement.

4.  **Pose Estimation and Visualization**: In parallel, the tool analyzes the thrower's form. It uses a pre-trained MediaPipe Pose Landmarker model (`pose_landmarker_lite.task`) to detect 33 distinct 3D landmarks on the thrower's body for a sequence of frames around the disc's release.

5.  **3D Wireframe Animation**: Finally, it uses the collected 3D landmark data to generate an interactive 3D wireframe animation of the throw using Matplotlib. This allows the user to view their throwing mechanics from any angle, providing invaluable feedback for form correction.

The repository is structured with a main script `process.py`, a `src` directory containing modular Python classes for each major task (`disc_tracker.py`, `perspective_calculator.py`, `pose_tracker.py`, `wireframe_animation.py`), and a `models` directory for the ML model. It leverages Python with libraries like OpenCV for image processing, NumPy for numerical calculations, MediaPipe for pose estimation, and Matplotlib for data visualization.

### Key Code and Structure Details

The project is well-structured, separating distinct functionalities into different classes within the `src/` directory.

**`process.py` - The Orchestrator**
This script is the main entry point. It uses `argparse` to handle command-line inputs like the video path, disc radius, and FPS. Its primary role is to manage the data flow between the different modules:
1.  It first loads all video frames into a NumPy array.
2.  It instantiates `PerspectiveCalculator` and feeds it the first half-second of frames to get the `ratio` (pixels-to-meters). This is done by averaging the ratios from multiple successful detections after removing outliers.
3.  It then splits the video frames, passing the second half (where the disc is flying) to the `DiscTracker`.
4.  After `DiscTracker` finds the disc, calculates speed/angle, and identifies the release frame (`frameIndex`), `process.py` takes a slice of frames around this point (`frames[frameIndex-2*args.fps:frameIndex+1*args.fps]`) for pose analysis.
5.  This slice is passed to `PoseTracker`, which extracts the 3D landmarks.
6.  Finally, the landmarks are passed to `WireframeAnimator` to generate the 3D animation.

**`src/perspective_calculator.py` - Camera Calibration**
This module's purpose is to find the conversion factor from pixels to real-world distance.
-   `process_frame()`: Takes a single frame, detects circles, and calculates the ratio.
-   `detect_disc()`: The core of this class. It converts the image to grayscale, blurs it, and then applies `cv2.HoughCircles`. This function is specifically tuned with `param1=200` and `param2=100` to robustly find disc-like objects.
-   `calculate_ratio(self, r)`: Implements the simple but effective formula `self.R / r` where `R` is the known physical radius and `r` is the detected pixel radius.

**`src/disc_tracker.py` - Flight Analysis**
This class is responsible for finding the disc in flight and calculating its speed.
-   `findBackground()`: A simple and effective method for static scenes. It computes a mean image across all frames in the provided video segment, which serves as a clean background plate.
-   `findDisc()`: This is the core tracking logic.
    -   It computes the absolute difference between the current frame and the background: `cv2.absdiff(image_blur, background)`.
    -   It then thresholds this difference (`cv2.threshold(absoluteDif, 90, 255, cv2.THRESH_BINARY)`) to create a binary mask where only the moving disc (and noise) is white.
    -   `cv2.findContours` is used on this mask to get the shape of the disc.
    -   `findExtrema` is a helper function that iterates through all points in the detected contours to find the min/max x and y values, defining a bounding box.
-   `findDiscSpeedAngle()`: This function takes the list of detected disc positions (rects). It calculates the distance between consecutive points and divides by the time delta (`1 / self.fps`) to get pixel speed. It multiplies by `pixelToRealRatio` for the final speed. It importantly uses `functions.remove_outliers` to clean both the speed and angle data before averaging, making the result more reliable.

**`src/pose_tracker.py` - Biomechanical Analysis**
This class acts as an interface to the MediaPipe pose landmarker model.
-   It initializes the MediaPipe `PoseLandmarker` using the `.task` model file.
-   `findKeypoints()`: This method iterates through the relevant video frames. For each frame, it performs necessary pre-processing (cropping, resizing to 256x256, converting to MediaPipe's `mp.Image` format) before passing it to `landmarker.detect()`. The results, which include normalized 3D coordinates (x, y, z) for 33 body landmarks, are stored.
-   `createWireFrame()`: This is a plotting utility for visualizing a *single* pose in 3D using `matplotlib`. It contains hardcoded connections between landmarks to draw a stick figure. For example, `betweenShoulders` is defined as `[(landmarks[11].x, landmarks[12].x), (landmarks[11].y, landmarks[12].y), (landmarks[11].z, landmarks[12].z)]`, which provides the start and end coordinates for drawing a line between the left and right shoulders.

**`src/wireframe_animation.py` - 3D Visualization**
This class is dedicated to creating the final 3D animation from the pose data.
-   It uses `matplotlib.animation.FuncAnimation` to create the animation object. `FuncAnimation` repeatedly calls an `update` function for each frame.
-   `update(self, frameIndex)`: This function is the core of the animation. For each `frameIndex`, it clears the 3D plot (`self.ax.cla()`), retrieves the corresponding landmark data from `self.landmarkedPoses`, and draws the 3D wireframe by plotting lines between the landmark coordinates, exactly like the static `createWireFrame` method. The key difference is that this is done repeatedly within the animation loop, creating the illusion of movement. This provides a powerful tool for reviewing the throwing motion from multiple perspectives.
