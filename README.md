# segment-anything-mini-annotator
A minimal annotator of image segmentation with Meta's Segment-Anything and Matplotlib in Python.

Everything is kept as minimal as possible using only matplotlib as GUI. 

 - Mouse "left" to record a point prompt,
 - Mouse "right" to delete the last clicked point. 
 - Press "ENTER" to generate a mask. 
 - Press "c" to clear all current annotations.
 - Press "x" to switch to "Background" mode, press again to switch back to "Foreground" mode.

## Requirements
Satisfy [Segment-Anything](https://github.com/facebookresearch/segment-anything) installation and you are good to go.