# STGCN model for ViEWS competition

## Hardware and software environment
The followings are the hardware environment of the author to conduct the experiment:
- CPU: Intel i9-9980XE CPU @ 3.00GHz
- GPU: Nviida Quadro RTX 8000
- Memory: 128GB

For Python environment, please run the following command:
```
conda env create --file base.yml
```
This line of command will read the base.yml file, and create a virtual environment named "base" with all required packages.

## Run the program
Please run the following command for train the model.
```
python src/main.py
```
This file automatically read the config file __task1.yaml__ from folder "src/configs/"
To run task 2 or task 3, please change line 36 in the main.py according.
