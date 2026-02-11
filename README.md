A method based on 6d pose estimation for surface point cloud collection of parts

## Usage
### Installation
```
pip install -r requirements.txt
cd sampling
python setup.py install
cd ..

My project.exe

```
## Result
### HandEyeCalibration
```
python HandEyeCalibration.py

# T_tool_cam=
# [[ 0.00001   0.999997  0.002265 -1.201593]
#  [-0.999998  0.000006  0.001771 29.310701]
#  [ 0.001771 -0.002265  0.999996 49.186424]
#  [ 0.        0.        0.        1.      ]]
```
### PointCloudCollection
```
python PointCloudCollection.py

# {4: 0.8081, 8: 0.8989, 12: 0.9318}
```
### PointCloudStitching
```
python PointCloudStitching.py

# {CD: 0.3262, HD: 1.7150}
```