
# Conda enviroment:

It's necessary to create a conda enviroment:
`conda create env name 'conda_-nameenv_name'`
`conda  activate -n 'conda_env_name'`

# Clothes-detection
Download the dataset and unzip it on the 'data' folder. Download link: https://universe.roboflow.com/thibauts-headquarters/clothes-detection-1kl0o.

# Make the inference process
Copy the absolute path directory and put it text on the 'image_path' variable content. Do it too with the model directory on the variable 'model_path'.

Then, run the command on the terminal:
`python3 src/val.py`