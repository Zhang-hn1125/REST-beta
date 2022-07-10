## Installation
Windows or Linux platform could both run.

pip install -r requirements.txt
```
## Training (Execute in the project folder. If you change a batch of data for training, please copy an empty project that has not tested data and execute it again.)
1.Generate the input particles
Using sub-tomograms avering method to rename the raw particles to 1_00****.mrc and place them into subtomo folder.

2.python process1.py

3.cd process2 folder 

4.Generate the ground truth of input particles
Generate the ground truth according to the paper introductions and place them into subtomo folder.

5.cd .. (Return to the previous directory)

6.python process3*(linux or windows).py

7.python process4.py

## Predict

1.Copy the input tomograms or sub-tomograms into tomoset folder and write the information for predicting in for_predict.star

2.Execute the command : python rest.py predict for_predict.star results/model_iter00.h5 --gpuID 0 --tomo_idx *(tomo_id)

