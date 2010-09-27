git clone git://github.com/vlfeat/vlfeat.git vlfeat
cd vlfeat/
make
cd ..
sudo mkdir /usr/local/include/vl
sudo cp ./vlfeat/vl/*.h /usr/local/include/vl/
sudo cp ./vlfeat/bin/glx/libvl.so /usr/local/lib/
sudo ldconfig
