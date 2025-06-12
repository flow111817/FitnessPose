# FitnessPose

**FitnessPose** is an automated posture analysis tool based on the **YOLOv8 Pose** model, focusing on precise posture recognition, standard evaluation, and motion imbalance detection of push up movements. Users can input video files (or use a camera for real-time analysis) by running the **app.py** script, and the system will parse actions frame by frame, outputting detailed evaluation metrics and visualization results.

## Started

### python

Suggest using Python virtual environment, which can be deployed with just one click using **Anaconda**

```bash
# Clone project to local
git clone https://github.com/flow111817/FitnessPose.git

# Enter the project directory
cd FitnessPose

# Create and activate a virtual environment (Conda is recommended)
conda create -n fitness_pose python=3.8
conda activate fitness_pose

# Install dependency packages
pip install -r requirements.txt

# Launch the application (supports video file or camera input)
python app.py
```

- If you haven't installed Conda, you can go to the Anaconda official website or use Miniconda lightweight deployment.
- This project relies on libraries such as **OpenCV**, YOLOv8 (from Ultralytics), **numpy**, etc. Using a virtual environment can effectively avoid version conflicts.
- The **PyTorch** version used in this project is 2.4.1, and it is recommended to download **version 11.8 of the Cudatoolkit** to fine tune the model using GPU

### Docker

```bash
# Pull the image
docker pull flow0817817/fitnesspose:latest

# Run
docker run -d -p 5000:5000 --name test flow0817817/fitnesspose:latest
```

## Contributing

Welcome and encourage contributions to **FitnessPose**.
