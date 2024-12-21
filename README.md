# Computer Vision Smart City [Link here!](https://huggingface.co/spaces/COS40007/Computer-Vision-Smart-City)
This project is primary used for detecting rubbish from image/video provided, using Yolov5 & Unet(not work) model.

# Overview
Our motivation for this project arises from the increasing need for automated solutions in urban waste management to keep growing cities clean and environmentally sustainable. In particular, illegal dumping and improper waste disposal are common problems in urban areas, impacting public health, safety, and aesthetics. Recognising this, our project aims to leverage
AI to identify and classify roadside rubbish using image data, thereby offering a smart city solution that could assist in more efficient waste management.

This AI model will primarily benefit local governments, waste management agencies, and environmental monitoring teams responsible for maintaining public spaces. These users will be able to detect rubbish quickly, distinguish between different types of waste, and use this information to prioritise areas that need immediate attention. By providing insights into waste patterns and disposal trends, our model can also help city councils make data-driven decisions that support sustainability efforts.

## Table of Contents
- [Pre-requisite](#pre-requisite)
- [Directory](#directory)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

<!--, best download Anaconda Navigator to create virtual environment easily-->
## Pre-requisites

Before you start, ensure you have the following:
- Python 3.8 or higher
- pip (Python package installer)
- [Git](https://git-scm.com/)
- [Anaconda](https://www.anaconda.com/) (optional for virtual environments)
- `streamlit` for the web-based UI


## Directory
```
root
┃📦model
┃┣ 📂unet
┃┃ ┗ 📜checkpoint_epoch5.pth
┃┣ 📂yolo
┃┃ ┗ 📜best.pt
┃┗ 📜.DS_Store
┃📦unet
┃┣ 📜__init__.py
┃┣ 📜unet_model.py
┃┗ 📜unet_parts.py
┃📦 yolov5
┃┣ 📂.git
┃┃ ┣ 📂hooks (Git hook samples)
┃┃ ┣ 📂info (Git metadata)
┃┃ ┣ 📂logs
┃┃ ┣ 📂objects
┃┃ ┣ 📂refs
┃┃ ┣ 📜HEAD
┃┃ ┣ 📜config
┃┃ ┗ ... (other Git system files)
┃┣ 📂.github
┃┃ ┣ 📂ISSUE_TEMPLATE (Issue templates: bug-report, feature-request, etc.)
┃┃ ┣ 📂workflows (CI/CD workflows: ci-testing.yml, docker.yml, etc.)
┃┃ ┗ 📜dependabot.yml
┃┣ 📂__pycache__ (Python bytecode files)
┃┣ 📂classify (Classification scripts: predict.py, train.py, val.py)
┃┣ 📂data
┃┃ ┣ 📂hyps (Hyperparameter YAMLs)
┃┃ ┣ 📂images (Sample images: bus.jpg, zidane.jpg)
┃┃ ┣ 📂scripts (Data fetching scripts: get_coco.sh, get_imagenet.sh, etc.)
┃┃ ┗ 📜*.yaml (Dataset configurations: coco.yaml, VOC.yaml, etc.)
┃┣ 📂models
┃┃ ┣ 📂hub (Model YAMLs: yolov5s.yaml, yolov5m.yaml, etc.)
┃┃ ┣ 📂segment (Segmentation model YAMLs)
┃┃ ┣ 📜*.py (Core model scripts)
┃┃ ┗ 📜*.yaml (Model configurations)
┃┣ 📂segment (Segmentation scripts: predict.py, train.py, val.py)
┃┣ 📂utils
┃┃ ┣ 📂aws (AWS deployment scripts)
┃┃ ┣ 📂docker (Docker configurations)
┃┃ ┣ 📂loggers (Logging utilities: wandb, comet, clearml)
┃┃ ┣ 📂segment (Segmentation utilities)
┃┃ ┗ 📜*.py (General utilities)
┃┣ 📜benchmarks.py
┃┣ 📜detect.py
┃┣ 📜export.py
┃┣ 📜hubconf.py
┃┣ 📜pyproject.toml
┃┣ 📜requirements.txt
┃┣ 📜train.py
┃┣ 📜tutorial.ipynb
┃┗ 📜val.py
┃
┣📜app.py
┣📜README.md
┗📜requirements.txt
```

---


## Installation

**In HTTPS mode**

1. **# Make sure you have git-lfs installed (https://git-lfs.com)**
```bash
   git lfs install
```

1. **Clone the Repository**
   ```bash
   git clone https://huggingface.co/spaces/COS40007/Computer-Vision-Smart-City
   cd Computer-Vision-Smart-City
   ```

For more, go to [Link](https://huggingface.co/spaces/COS40007/Computer-Vision-Smart-City/tree/main?clone=true) to have more details on HTTPS, even SSH

2. **Set Up a Virtual Environment**
   
    Using Anaconda (recommended):
   ```bash
   conda create -n yolo-unet-env python=3.8
   conda activate yolo-unet-env
 
   ```

   Using `venv`:
   ```bash
   python3 -m venv env
   source env/bin/activate  # On Windows: `env\Scripts\activate`
   ```
   
   Or we can download the Anaconda Navigator to easy install virtual environment

3. **Install Dependencies**
   ```bash
   pip install -r path_to/requirements.txt
   ```

    Please copy the full path at requirements.txt at root directory then replace the `path_to/requirements.txt` to ensure install easily. 

---

## Training

### YOLOv5 Training

Navigate to the `yolov5` directory and start training:
```bash
cd yolov5
python train.py --data data.yaml --cfg yolov5s.yaml --weights yolov5s.pt --epochs 100
```

### UNet Training

Navigate to the `unet` directory and start training:
```bash
cd ../unet
python train_unet.py --data data/train/ --epochs 50 --batch-size 16
```

---

## Running the Streamlit App

To run the Streamlit UI for model interaction and visualization:

1. Navigate to the project root directory.
2. Run the following command:
   ```bash
   streamlit run app.py
   ```
3. Open your browser and go to the URL provided by Streamlit (usually `http://localhost:8501`).

### Features of the App
- Upload image/video to test the trained YOLOv5 or UNet models.
- Choice to detect only rubbish (None) or whatever you want based on the list that displays, or even all at textbox to detect all.
- Visualize YOLOv5 object detection results.
- Segment images using the UNet model.

---

## License

This project is licensed under the [MIT License](LICENSE).

---
