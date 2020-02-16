
rm -rf build
mkdir build
cd build
# cmake .. -DCMAKE_INSTALL_PREFIX=/share/gnuradio/grnext/ \
#  -DENABLE_STATIC_LIBS=False \
#  -DENABLE_DOXYGEN=OFF \
#  -DENABLE_GR_AUDIO=OFF \
#  -DENABLE_GR_ANALOG=ON \
#  -DENABLE_GNURADIO_RUNTIME=ON \
#  -DENABLE_GR_BLOCKS=ON \
#  -DENABLE_GR_DIGITAL=ON \
#  -DENABLE_GR_FEC=OFF \
#  -DENABLE_GR_FFT=ON \
#  -DENABLE_GR_DTV=OFF \
#  -DENABLE_GR_TRELLIS=OFF \
#  -DENABLE_GR_FILTER=ON \
#  -DENABLE_GR_QTGUI=OFF \
#  -DENABLE_GR_UHD=OFF \
#  -DENABLE_PYTHON=ON \
#  -DENABLE_VOLK=ON \
#  -DENABLE_GRC=OFF \
#  -DENABLE_GR_VOCODER=OFF \
#  -DENABLE_GR_CTRLPORT=OFF \
#  -DENABLE_GR_WAVELET=OFF \
#  -DENABLE_GR_UTILS=ON \
#  -DENABLE_GR_VIDEO_SDL=OFF \
#  -DENABLE_TESTING=ON \
#  -DENABLE_GR_ZEROMQ=OFF \
#  -DENABLE_GR_MODTOOL=ON \
#  -DENABLE_GR_BLOCKTOOL=ON \
#  -DCMAKE_BUILD_TYPE=RelWithDebInfo
# #  -DCMAKE_BUILD_TYPE=NoOptWithASM 
# #  -DCMAKE_CXX_FLAGS="-H"

cmake .. -DCMAKE_INSTALL_PREFIX=/share/gnuradio/grnext/ \
 -DENABLE_STATIC_LIBS=False \
 -DENABLE_DOXYGEN=OFF \
 -DENABLE_GR_AUDIO=OFF \
 -DENABLE_GR_ANALOG=OFF \
 -DENABLE_GNURADIO_RUNTIME=ON \
 -DENABLE_GR_BLOCKS=ON \
 -DENABLE_GR_DIGITAL=OFF \
 -DENABLE_GR_FEC=OFF \
 -DENABLE_GR_FFT=OFF \
 -DENABLE_GR_DTV=OFF \
 -DENABLE_GR_TRELLIS=OFF \
 -DENABLE_GR_FILTER=OFF \
 -DENABLE_GR_QTGUI=OFF \
 -DENABLE_GR_UHD=OFF \
 -DENABLE_PYTHON=ON \
 -DENABLE_VOLK=ON \
 -DENABLE_GRC=OFF \
 -DENABLE_GR_VOCODER=OFF \
 -DENABLE_GR_CTRLPORT=OFF \
 -DENABLE_GR_WAVELET=OFF \
 -DENABLE_GR_UTILS=ON \
 -DENABLE_GR_VIDEO_SDL=OFF \
 -DENABLE_TESTING=ON \
 -DENABLE_GR_ZEROMQ=OFF \
 -DENABLE_GR_MODTOOL=ON \
 -DENABLE_GR_BLOCKTOOL=ON \
 -DCMAKE_BUILD_TYPE=NoOptWithASM
#  -DCMAKE_BUILD_TYPE=NoOptWithASM 
#  -DCMAKE_CXX_FLAGS="-H"

/usr/bin/time --verbose make -j8
# make -j8
