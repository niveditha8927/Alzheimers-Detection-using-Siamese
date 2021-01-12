# Alzheimers-Detection-using-Siamese
Implementation of paper - Alzheimer's disease diagnosis from structural MRI using Siamese convolutional neural network

M. Amin-Naji, H. Mahdavinataj and A. Aghagolzadeh, "Alzheimer's disease diagnosis from structural MRI using Siamese convolutional neural network," 2019 4th International Conference on Pattern Recognition and Image Analysis (IPRIA), Tehran, Iran, 2019, pp. 75-79, doi: 10.1109/PRIA.2019.8786031.

Differences from the paper -
Used a combination of OASIS1 and OASIS3 datasets
With the model which works on 2D slices of MRI scan obtained highest accuracy of 95.67%

Train_res_fit.py is the training file. 

Note:
Keep the triplets generation file - gen_pairs_oasis and other files in the same directory.

Preprocessing of raw MRI scans to smoothed Gray Matter volume maps done as shown in the figure - 
https://www.youtube.com/watch?v=YVDG9cjnUPU&feature=youtu.be

Final volume map is sliced into 2D slices which are used grouped into Triplets and used in the network. Volume map breaking into slices and Triplets generated in the file gen_pairs_oasis.py.

