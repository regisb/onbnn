cd lshkit-0.2.1/
sudo cp FindGSL.cmake /usr/share/cmake-2.8/Modules/
mkdir build
cd build
cmake ..
make
sudo mkdir /usr/local/include/lshkit
sudo cp -r ../include/* /usr/local/include/lshkit/
sudo cp lib/liblshkit.a /usr/local/lib
cd ../..
