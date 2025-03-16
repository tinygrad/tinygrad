ARCH=sm_80
.PHONY: clean
clean:
	-rm *.cuasm *.cubin *.ptx
	-rm addHost
	
%Host: %Host.cu
	nvcc -arch sm_80 -o ${@} ${<} -lcuda	-lcublas

%.ptx: %.cu
	nvcc -o ${@} --ptx ${<}

dumped.%.cubin: %.ptx
	ptxas -arch=${ARCH} -m64 ${<} -o ${@}

%.cuasm: dumped.%.cubin
	python dump.py -i ${<} -o ${@}
	
%.cubin: %.cuasm
	python assemble.py -i ${<} -o ${@}
	

