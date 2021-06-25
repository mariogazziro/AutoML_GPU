all:
	g++  --std=c++11 -Ofast sparse_mat.cc -o sparse_mat_cpu
	nvcc --std=c++11 -O3    sparse_mat.cu -o sparse_mat_gpu
	./run
