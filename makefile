main: cpu.o gpu.o
	g++ cpu.o `pkg-config opencv --cflags --libs` -o cpu
	nvcc `pkg-config opencv --cflags --libs` -I/usr/include/opencv2 gpu.o -o gpu

cpu.o: main.cpp
	g++ -c main.cpp `pkg-config opencv --cflags --libs` -o cpu.o

gpu.o: main.cu
	nvcc `pkg-config opencv --cflags --libs` -I/usr/include/opencv2 -c main.cu -o gpu.o

clean:
	rm gpu.o cpu.o cpu gpu output/*