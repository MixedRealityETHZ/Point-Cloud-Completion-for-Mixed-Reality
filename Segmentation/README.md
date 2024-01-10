### Code to segment hand point clouds from entire poinct cloud by using RGB and depth data and SAM model

## install SAM into environment: 
# pip install segment-anything

## Place RGB and Depth data with corresponidng meta data from HoloLens recording into the defined path:
# adjust "sourcepath" in PointCloudGeneration.py, line 221
# place data in sourcepath/ImageCloud/depth_ply and sourcepath/ImageCloud/rgb_ply

## Run code:
# python PointCloudCompletion
# Segmented data is saved into sourcepath/ImageCloud/colored_ply
