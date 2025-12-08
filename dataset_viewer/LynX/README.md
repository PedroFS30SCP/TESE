## Lynx Dataset Summary

**Overview**  
A large collection of hand‐gesture recordings captured in diverse indoor and outdoor scenarios using an event‐based GenX320 camera. Each gesture clip is temporally segmented and labeled at the frame level, enabling precise study of hand‐movement duration, position, and timing.

**Acquisition Details**  
- **Event‐Based Recording**  
  - Frames are triggered only when significant pixel‐level changes occur  
  - Reduces redundancy, lowers storage needs, and naturally segments gestures  
  - Captures meaningful motion events without superfluous frames  
- **GenX320 Sensor**  
  - High temporal resolution (20 fps equivalent for segmentation) and low latency  
  - Wide dynamic range, low power consumption, and pixel‐level change detection  
  - Ideal for accurately capturing fast hand movements in both bright and dim settings  



## Dataset Statistics

### Global Information
- Total duration: 4.10 hours
- Total number of subjects: 18
- Total number of analyzed gestures: 13
- Total number of gesture executions: 7,577

### Duration by Scenario
- Indoor Dynamic (scenario_i_d): 0.99 hours
- Indoor Static (scenario_i_s): 1.03 hours
- Indoor Window (scenario_i_w): 0.95 hours
- Outdoor Dynamic (scenario_o_d): 1.13 hours

## Gesture Types and Statistics

The following tables report gesture statistics, clearly separated into dynamic gestures (first table) and static gestures (second table).

#Dynamic Gestures:

| Gesture        | Count | Mean (s) | Std. Dev. (s) | Median (s) | 95% CI (s)     |
|----------------|-------|----------|---------------|------------|----------------|
| Swipe Left     | 652   | 0.59     | 0.19          | 0.55       | [0.58, 0.61]   |
| Swipe Right    | 700   | 0.61     | 0.20          | 0.55       | [0.59, 0.62]   |
| Swipe Up       | 703   | 0.61     | 0.20          | 0.55       | [0.60, 0.63]   |
| Swipe Down     | 659   | 0.62     | 0.23          | 0.55       | [0.60, 0.64]   |
| Zoom In        | 746   | 0.60     | 0.24          | 0.55       | [0.58, 0.61]   |
| Zoom Out       | 722   | 0.69     | 0.27          | 0.65       | [0.67, 0.71]   |
| Select         | 868   | 0.51     | 0.22          | 0.45       | [0.50, 0.52]   |
| Double Tap     | 813   | 0.76     | 0.28          | 0.70       | [0.74, 0.78]   |
| Rotate CW      | 716   | 0.95     | 0.38          | 0.90       | [0.92, 0.98]   |
| Rotate CCW     | 757   | 0.90     | 0.36          | 0.85       | [0.88, 0.93]   |
| Screenshot     | 643   | 0.69     | 0.29          | 0.65       | [0.66, 0.71]   |

#Static Gestures:

| Gesture   | Count | Mean (s) | Std. Dev. (s) | Median (s) | 95% CI (s)     |
|-----------|-------|----------|---------------|------------|----------------|
| Thumb Up  | 98    | 12.13    | 8.49          | 14.78      | [10.45, 13.81] |
| Clap      | 100   | 12.09    | 8.06          | 13.73      | [10.51, 13.67] |

## Dataset Structure

```
ROOT_DIR/
├── subject_X/
│   ├── scenario_Y/
│   │   ├── gesture_Z/
│   │   │   ├── frames/ (or images/)
│   │   │   │   ├── 1.png
│   │   │   │   ├── 2.png
│   │   │   │   └── ...
│   │   │   └── annotations/
│   │   │       ├── 1.txt
│   │   │       ├── 2.txt
│   │   │       └── ...
│   │   └── ...
│   └── ...
└── ...
```

### Key Components

- **Subjects (subject_X)**: Different participants performing the gestures
- **Scenarios (scenario_Y)**: Various contexts or situations
  - i_d: Indoor Dynamic conditions
  - i_s: Indoor Static conditions
  - i_w: Indoor Window conditions
  - o_d: Outdoor Dynamic conditions
- **Gestures (gesture_Z)**: Specific gesture sequences
- **Frames**: Individual video frame images
- **Annotations**: YOLO format text files containing bounding box coordinates

## Annotation Format

Annotations are provided in YOLO format, where each .txt file contains:
- Gesture ID
- Bounding box center coordinates
- Bounding box dimensions (width and height)


## Segmentation CSV File

A CSV file (`gesture_segmentation.csv`) accompanies the dataset. Each row corresponds to a single gesture instance with the following columns:

- **subject**: Participant identifier (e.g., `subject_1`, `subject_12`)  
- **scenario**: Recording environment (e.g., `scenario_i_s` for Indoor Static, `scenario_o_d` for Outdoor Dynamic)  
- **gesture_type**/**gesture_sequence**: Numeric label (integer) representing the gesture category  (or gesture_X where x is the gesture_type)
- **gesture_name**: Human‐readable gesture label (e.g., `Swipe Left`, `Thumb Up`)  
- **start_frame**: Index of the first frame where the gesture begins (zero‐based)  
- **end_frame**: Index of the last frame where the gesture ends  
- **num_frames**: Total number of frames spanning the gesture (`end_frame – start_frame + 1`)  
- **duration_seconds**: Real‐time duration (in seconds) of the gesture (`num_frames / fps`)  
- **fps**: Frame rate used for segmentation (constant at 20.0)  
- **video_length_frames**: Total number of frames in the source video clip  

By using this CSV, you can easily filter, sort, or aggregate gestures by subject, scenario, or duration. For example, to load and inspect the first few entries:

## License

The MIT License (MIT)
=====================

Copyright © <2025> <Politecnico di Milano and ETH Zürich>

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the “Software”), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
