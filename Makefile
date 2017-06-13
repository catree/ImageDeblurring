all :
	g++  -g -w -I".\opencv\build\install\include" -L".\opencv\build\install\x64\mingw\lib" deblurring.cpp -lopencv_core320 -lopencv_highgui320 -lopencv_imgproc320 -lopencv_imgcodecs320