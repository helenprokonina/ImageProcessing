# ImageProcessing

Here there are four files:
* <b>HW_1.ipynb</b> - Jupyter notebook where all steps for image alignment and pan-sharpening are performed.
* <b>HW_1_script.py</b> - python script where everything from the previous file is automated.
* <b>utils.py</b> - script with alignment and pan-sharpening functions
* <b>CameraCalibration.ipynb</b> - notebook for aditional task

<br>In <b>colored</b> folder there are images for alignment: RGB_half.JPG and RGB_quater.JPG</br>
<br>In <b>pan</b> folder there is original panochromic image of high quality.</br>
<br>In <b>results</b> folder there are results from alignment and from various pan-sharpening methods (Brovey, Esri and Simple mean).</br>
<br>In <b>chess</b> folder there are images of chess for camera calibration.</br>
<br>In <b>calibrated</b> folder there are results of calibration and 3d-rendering task.</br>

## HW_1_script.py

To run this script print the following:

```bash
python HW_1_script.py -i colored -t pan
```
where i - input folder with colored images, t - target folder with panochromic image.
