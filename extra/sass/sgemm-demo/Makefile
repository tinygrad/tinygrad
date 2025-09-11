ARCH=sm_86

clean:
	rm matmul.cubin matmulHost
	
%Host: %Host.cu
	nvcc -arch ${ARCH} -o ${@} ${<} -lcuda	-lcublas

%.cubin:
	python assemble.py -o matmul.cubin
	

