#!/bin/bash
pip3 install virtualenv
virtualenv -p /usr/bin/python3 volume_estimation
cd ./volume_estimation
source ./bin/activate
cp -r ../vol_estimation_pipeline ./
pip3 install opencv-python
git clone https://github.com/OpenKinect/libfreenect
cd libfreenect
mkdir build
cd build
cmake -L .. # -L lists all the project options
sudo make install
sudo ldconfig /usr/local/lib64/
sudo adduser $USER video
sudo adduser $USER plugdev
sudo echo '# ATTR{product}=="Xbox NUI Motor"
SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="02b0", MODE="0666"
# ATTR{product}=="Xbox NUI Audio"
SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="02ad", MODE="0666"
# ATTR{product}=="Xbox NUI Camera"
SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="02ae", MODE="0666"
# ATTR{product}=="Xbox NUI Motor"
SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="02c2", MODE="0666"
# ATTR{product}=="Xbox NUI Motor"
SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="02be", MODE="0666"
# ATTR{product}=="Xbox NUI Motor"
SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="02bf", MODE="0666"' >> /etc/udev/rules.d/51-kinect.rules

sudo apt-get install cython3
sudo apt-get install python3-dev
sudo apt-get install python3-numpy

cd ../wrappers/python
python3 setup.py build_ext --inplace

cp freenect.cpython-38-x86_64-linux-gnu.so ../../../vol_estimation_pipeline
# cd ../../../vol_estimation_pipeline

pip3 install open3d
pip3 install numpy
pip3 install tensorflow
pip3 install pyserial
pip3 install flask
pip3 install Flask-SocketIO
pip3 install h5py

# git clone https://github.com/singhals163/GrowSimpleInterIIT.git
