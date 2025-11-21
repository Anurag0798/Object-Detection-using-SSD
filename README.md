# Object Detection using SSD

A practical real-time object detection system based on SSD (Single Shot MultiBox Detector) architecture using OpenCV's DNN module. This project demonstrates how to detect multiple object classes - including vehicles, people, and more in-driving footage and visualize results along with lane detection for a richer autonomous driving simulation.

## Features

- **Real-Time Object Detection:**
  Uses MobileNet-SSD pre-trained model (Caffe) to detect 20+ common object types in images or video frames.
- **Simultaneous Lane Detection:**
  Integrates custom lane detection via classic computer vision to overlay detected driving lanes.
- **Combined Visualization:**
  Shows both lane lines and object bounding boxes on each processed frame for a realistic driving environment augmentation.
- **Modular & Extensible:**
  Code is organized for easy adaptation: swap the model, tweak classes, thresholds, or improve lane detection as needed.

## Repository Structure

```
.
├── ssd_object_lane_detection.py   # Main script: runs SSD detection + lane marking on video
├── models/
│   ├── deploy.prototxt            # SSD MobileNet Caffe config file (download separately)
│   ├── mobilenet_iter_73000.caffemodel # SSD MobileNet Caffe weights (download separately)
│   ├── nD_7.mp4                   # (Sample driving video; replace with your own as needed)
│   └── lane_utils.py              # Lane detection helper functions
└── README.md                      # This file
```

---

## Requirements

- Python 3.7+
- opencv-python
- numpy

Install all requirements:
```bash
pip install opencv-python numpy
```
## Setup & Usage

1. **Clone the repository:**
    ```bash
    git clone https://github.com/Anurag0798/Object-Detection-using-SSD.git

    cd Object-Detection-using-SSD
    ```

2. **Download the model weights and config:**
    - Ensure `models/deploy.prototxt` and `models/mobilenet_iter_73000.caffemodel` are present.  

3. **Prepare your video:**
    - Place your test video file in `models/nD_7.mp4` (or update the script to point to your own video).

4. **Run the detection + lane marking:**
    ```bash
    python ssd_object_lane_detection.py
    ```
    - Frames will be shown with both object and lane overlays.
    - Press `q` in the window to exit at any time.

## How It Works

- **ssd_object_lane_detection.py:**
  - Loads the SSD MobileNet model using OpenCV DNN.
  - For each video frame:
      - Applies lane detection via `detect_lane()` (from `lane_utils.py`).
      - Detects objects, drawing bounding boxes and class labels (confidence threshold ≥ 0.3).
      - Merges and displays results in one visualization window.
  - Designed for easy extension to new classes, color schemes, or lane algorithms.

- **lane_utils.py:**
  - Provides utilities for grayscale conversion, edge detection, masking, and Hough Line Transform to detect lane lines.

## Object Classes Detected

The SSD model recognizes:
`aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor`

## Customization

- **Use a different video:**  
  Change the path in `cap = cv2.VideoCapture("models/nD_7.mp4")`.
- **Change detection threshold or classes:**  
  Tweak confidence in `if confidence > 0.3:`; customize the `CLASSES` list.
- **Expand lane detection:**  
  Modify `lane_utils.py` to change mask shape, line color, or processing parameters.

## License

This repository is for educational and research purposes.  
LICENSE file added for more details.

## Contributions

Contributions & suggestions welcome!  
Feel free to fork, adapt, and submit improvements via pull request.  
If you use or learned from this project, a star is appreciated!