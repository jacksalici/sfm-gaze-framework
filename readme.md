# SFM-based gaze estimation framework 🎥👀

> This work has been developed as part of my Master's Degree Thesis title **"Leveraging Gaze Estimation in Human-Robot Interaction: development of a Framework for Eye Tracking in the Wild"**, under the supervision of Prof. R. Vezzani. 🎓

![Example of the framework output](https://raw.githubusercontent.com/jacksalici/sfm-gaze-estimation-framework/main/imgs/fw-example.png)

## Abstract 📜

Humanity has always developed tools and skills to simplify tasks and improve the quality of life. Since the invention of robotic arms, these machines have reduced the physical demands of labor-intensive jobs and increased safety. With the advent of artificial intelligence, robots are becoming more empathetic and interactive, making a robust human-robot interaction (HRI) essential. A key aspect of interactive robots is their ability to detect users’ intention to engage and interpret their focus of attention -- tasks where estimating human gaze plays a crucial role.

The goal of this thesis is to develop a framework for gaze estimation and eye tracking. Firstly, it addresses the practical challenge of estimating eye movements “in the wild”. Secondly, it provides a baseline for the future research in this field. The framework makes use of Meta’s Project Aria glasses, a device designed to accelerate research in augmented and extended reality (AR/XR), and integrates multiple technologies, from neural networks to Structure From Motion (SFM) processing. The gaze direction can effectually be localized within a 3D environment, allowing tracking across multiple third-person perspectives.

A new dataset has also been recorded to demonstrate the framework’s capabilities and offer a potential benchmark and validation tool for new gaze estimation models, especially in HRI context. It consists of recordings of participants performing common actions while wearing Project Aria glasses, captured from another pair of glasses and a Pepper robot the participant interacts with. This dataset could address existing challenges in the field and advance further research in HRI.

_👉 The full thesis is available at [this link](https://morethesis.unimore.it/theses/available/etd-09262024-135659/)_

## Pipeline 🚥
The framework is composed of several Python scripts that work together to process the data and estimate gaze direction. The main steps are as follows:

1. `recordManager.py` to download all the files and save them.
2. `timestampDetector.py` to detect QR code starts and ends as well as synced video pairs and save them in a `csv` file.
3. `frameExtractor.py` to save frames and the gaze 2d info from video using the just found timestamps.  
4. `reconstruction.py` to made the SFM computation.  
    - _[optional]_ `localize.py`, localize a image in the SFM model.
5. `gazeTo3dpoints.py` to convert the gaze direction to the 3d gaze position.
6. `rerunLogger.py` for the ReRun visualization.  

### Requirements 📦
- Python 3.8+, Numpy, OpenCV
- HLOC and PyColMap for the SFM reconstruction
- ReRun and Plotly for the visualization

Moreover, the framework uses [`aria-glasses-utils`](https://github.com/jacksalici/aria-glasses-utils) a wrapper library for the Project Aria glasses, which provides handy functions to download and process the data.

## Other Resources 🔗
- `src/qrcodeServer`: Flask server to generate QR codes for the timestamps.
- `src/timeStampApp`: Android app developed using Mit App Inventor to generate timestamps for manual video synchronization.
- `src/external`: Third-party libraries for TicSync protocol and other utilities.

## Dataset 📂
The framework has been tested recording a dataset for gaze estimation in HRI context. It consists of 10 participants performing common actions while wearing Project Aria glasses, captured from another pair of glasses and a Pepper robot the participant interacts with. For more information, please contact me.