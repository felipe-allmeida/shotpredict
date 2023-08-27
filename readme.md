Please install python and pytorch in your macos
https://pytorch.org/get-started/locally/
https://towardsdatascience.com/installing-pytorch-on-apple-m1-chip-with-gpu-acceleration-3351dc44d67c
https://www.mrdbourke.com/pytorch-apple-silicon/

you can pick one of them to install package

then you have to install opencv
pip install opencv-python

you can execute the script
python shotpredict.py

output.json does not store all the results. 
it is overwritten by the data coming. it means that it only store current statistics.


you can modify script to use different video as input. but only tobin videos are ok because hoop position.
