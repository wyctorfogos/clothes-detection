
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

## API access
It's necessary to change the model folder path in 'src/models/object_detection.py'. Then run the command bellow:
`python3 src/app.py`

Now it's possible to get the clothe's information using a simple REST api call. It's running on the port '8000' by default, but can be changed.

Its a POST request typy to 'clothes_detection', where a file is necessary to be indexed with a name 'image' on the form-data.

![Clothes detection](https://github.com/wyctorfogos/clothes-detection/blob/main/ClothesDetection.png)
