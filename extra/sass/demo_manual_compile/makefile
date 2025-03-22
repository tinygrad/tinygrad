ARCH=sm_86
.PHONY: clean
clean:
	-rm *.cuasm *.cubin *.ptx
	-rm addHost
	
%Host: %Host.cu
	nvcc -arch ${ARCH} -o ${@} ${<} -lcuda	-lcublas

%.ptx: %.cu
	nvcc -arch ${ARCH} -o ${@} --ptx ${<}

dumped.%.cubin: %.ptx
	ptxas -arch ${ARCH} -m64 ${<} -o ${@}

%.cuasm: dumped.%.cubin
	python dump.py -i ${<} -o ${@}
	
%.cubin: %.cuasm
	python assemble.py -i ${<} -o ${@}
	

