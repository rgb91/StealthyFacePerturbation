# Stealthy Perturbations of Facial Images by Spatial Transformation of Face-features
Project for CS6216 Sem 2 AY2018/19 Project | NUS (Singapore)

**Sanjay Saha (E0321282)** 

Email: [sanjaysaha@comp.nus.edu.sg](mailto:sanjaysaha@comp.nus.edu.sg)
___
### Prerequisite Files:
Inception ResNet v1 Model (trained on CASIA-WebFace) and Shape predictor model [Link](https://drive.google.com/drive/folders/10jCd8qaH-z3zhYNVoT_E_1uND6ERoEWk?usp=sharing)
___
### Run Instruction:
At first, download the models from the given link above. Execute the following command from terminal. You need to `cd` into the project directory first.
```
python run.py
--pretrained_model "path to the Inception ResNet v1 model trained on CASIA-WebFace" \
--dlib_model "path to the shape predictor model" \
--img "path to the input image" \
--true_label "label of the input image" \
--output_dir "path to the directory to save results"
```
 