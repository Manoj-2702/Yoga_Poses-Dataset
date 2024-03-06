# TDL-Dataset
 
The code utilizes the MediaPipe library for pose estimation, which provides a pre-trained pose detection model. It processes images containing individuals performing yoga poses and extracts the pose landmarks, which are then used to calculate the angles between different body joints.

The dataset includes the following information for each yoga pose image:

- <b>Label</b>: Name of the yoga pose (filename).
- <b>Left Elbow Angle</b>: Angle between the left shoulder, elbow, and wrist joints.
- <b>Right Elbow Angle</b>: Angle between the right shoulder, elbow, and wrist joints.
- <b>Left Shoulder Angle</b>: Angle between the left elbow, shoulder, and hip joints.
- <b>Right Shoulder Angle</b>: Angle between the right elbow, shoulder, and hip joints.
- <b>Left Knee Angle</b>: Angle between the left hip, knee, and ankle joints.
- <b>Right Knee Angle</b>: Angle between the right hip, knee, and ankle joints.

## Setup

- Clone the repository:
```
git clone https://github.com/Manoj-2702/TDL-Dataset.git
```
- Navigate to the project directory:
```
cd TDL-Dataset
```
- Install the required dependencies:
```
pip install -r requirements.txt
```
- Add images to the required pose to the TRAIN Folder and the respective pose folder


## Usage
- Ensure that your yoga pose images are stored in the TRAIN/ directory within the project folder.

- Run the script main.py to process the images and collect the pose data:
```
python collect_data.py
```

## Results
There will be two datasets which will be created. One containing the labels and the angles calculated. The other dataset contains the mediapipe landmarks for all the images.

## Contributing
Contributions to this project are welcome. If you encounter any issues or have suggestions for improvements, feel free to open an issue or create a pull request.

