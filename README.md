# Yoga-Poses-Dataset
This project was done as a part of the data collection project of Topics in Deep Learning(TDL).

The Yoga-Pose Dataset is a comprehensive collection of yoga pose images, pose landmarks, and calculated angles for various yoga poses. This dataset serves as a valuable resource for researchers, practitioners, and developers interested in yoga pose detection, recognition, and analysis. Each yoga pose in the dataset is meticulously annotated with pose landmarks and corresponding angles, providing rich ground truth data for machine learning model training and evaluation.

The code utilizes the MediaPipe library for pose estimation, which provides a pre-trained pose detection model. It processes images containing individuals performing yoga poses and extracts the pose landmarks, which are then used to calculate the angles between different body joints.

## Dataset Contents

The dataset consists of the following components:

- <b>Images</b>: High-quality images of different yoga poses captured from various perspectives and angles.

- <b>Pose Landmarks</b>: Pose landmarks extracted using the MediaPipe Pose model, providing key points representing different body parts in each yoga pose image.

- <b>3D Models</b>: Three-dimensional (3D) models generated from the pose landmarks, allowing visualization of the yoga poses in a three-dimensional space.

- <b>Calculated Angles</b>: Angles between specific body parts in each yoga pose, providing detailed insights into the pose dynamics and alignment.

### Yoga Poses Included

The dataset covers a wide range of yoga poses, including:

1. <b>ArdhaChandrasana (Half-Moon)</b>

![v12](https://github.com/Manoj-2702/TDL-Dataset/assets/103581128/52f55bf1-f2fd-4629-b355-c9b49cef6072)

2. <b>Downward Dog</b>

![downward_dog17](https://github.com/Manoj-2702/TDL-Dataset/assets/103581128/432dc745-7a3e-45c2-9780-ed3949ea8ace)

3. <b>Triangle</b>

![triangle25](https://github.com/Manoj-2702/TDL-Dataset/assets/103581128/a60cd04e-e24b-4e88-b305-482754041136)

4. <b>Veerabhadrasana</b>

![Veerabhadrasana_30](https://github.com/Manoj-2702/TDL-Dataset/assets/103581128/724d8d76-5267-4448-9e55-d021d5dc637e)

5. <b>Natarajasana</b>

![download](https://github.com/Manoj-2702/TDL-Dataset/assets/103581128/87fe37a5-7350-4c54-b30e-f19fe84ea12a)

6. <b>Vrukshasana</b>

![v59](https://github.com/Manoj-2702/TDL-Dataset/assets/103581128/4fd8b74d-4668-4e06-b9c8-8ac6179742c3)

7. <b>BaddhaKonasana</b>

![BK_6](https://github.com/Manoj-2702/TDL-Dataset/assets/103581128/44a2f19a-2bc2-4458-8aa4-03c420a9154e)

8. <b>UtkataKonasana</b>

![UK_5](https://github.com/Manoj-2702/TDL-Dataset/assets/103581128/52484a06-559f-4d4d-a4b3-eaa949782729)


### Calculated Angles

1. Elbow Angles

```
  Left Elbow Angle: Angle between the left shoulder, elbow, and wrist points.

  Right Elbow Angle: Angle between the right shoulder, elbow, and wrist points.
```

2. Shoulder Angles

```
  Left Shoulder Angle: Angle between the left elbow, shoulder, and hip points.

  Right Shoulder Angle: Angle between the right hip, shoulder, and elbow points.
```

3. Knee Angles

```
  Left Knee Angle: Angle between the left hip, knee, and ankle points.

  Right Knee Angle: Angle between the right hip, knee, and ankle points.
```

4. Additional Angles

```
  Angle for ArdhaChandrasana 1: Angle specific to ArdhaChandrasana, calculated between ankle, hip, and opposite ankle points.

  Angle for ArdhaChandrasana 2: Second angle specific to ArdhaChandrasana, calculated between ankle, hip, and opposite ankle points.

  Hand Angle: Angle between the left elbow, right shoulder, and right elbow points.

  Left Hip Angle: Angle between the left shoulder, left hip, and left knee points.

  Right Hip Angle: Angle between the right shoulder, right hip, and right knee points.
```

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
python main.py
```

## USE

This dataset is used for training deep learning models for detecting yoga poses from an image.
Researchers, developers, and practitioners can utilize this dataset for various purposes, including:

- Training and evaluating machine learning models for yoga pose detection and recognition.
- Conducting experiments and studies on yoga pose dynamics and alignment.
- Developing applications and tools for yoga practitioners to improve their practice.

## Results

There will be two datasets which will be created. One containing the labels and the angles calculated. The other dataset contains the mediapipe landmarks for all the images.

## Contributing

Contributions to this project are welcome. If you encounter any issues or have suggestions for improvements, feel free to open an issue or create a pull request.
