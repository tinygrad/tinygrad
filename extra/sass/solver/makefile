ARCH=sm_89
LIBS := libcublasLt.txt libcublas.txt

find_%:
	find /usr/local/cuda/ -name "*${*}*"
	
stdlibs/%.so:
	mkdir -p stdlibs
	cp /usr/local/cuda/targets/x86_64-linux/lib/${*}.so $@

tinygrad_gen/%:
	mkdir -p tinygrad_gen && \
	export DUMP_CUDA_DIR=$$(pwd)/tinygrad_gen && \
	export DUMP_CUDA_PREFIX=${*} && \
	cd ../../.. && \
	export PYTHONPATH="." && \
	NV=1 python test/${*}.py

# do nvcc --ptx -arch ${ARCH} -o ${@}/$${cufile} $${cufile}; \
	
tinygrad_gen/%.txt:
	export dir=tinygrad_gen;\
	export PYTHONPATH="/home/alvy/gbin/tinygrad"; \
	for cufile in tinygrad_gen/${*}_*.cu; \
		do \
			echo $${cufile};\
			filename_ext=$$(basename -- $${cufile}); \
			base="$${filename_ext%.*}"; \
			nvcc --ptx -arch ${ARCH} -o $${dir}/$${base}.ptx $${cufile}; \
			ptxas -arch=${ARCH} -m64 -o $${dir}/$${base}.cubin $${dir}/$${base}.ptx; \
			python disasm.py -a ${ARCH} -o $${dir}/$${base}.sass $${dir}/$${base}.cubin; \
			python solve.py -a ${ARCH} -s $${dir}/$${base}.sass -d $${dir}/$${base}.txt; \
		done; \
	python merge.py ${*}.txt $${dir}/*.txt

%.sass: cubins/%.cubin
	python disasm.py -a ${ARCH} ${<} -o ${@}

%.sass: stdlibs/%.so 
	python disasm.py -a ${ARCH} ${<} -o ${@}

%.txt: %.sass
	python solve.py -s ${<} -d ${@} -a ${ARCH}
	
backup_%:
	cp ../assembler/InsAsmRepos/DefaultInsAsmRepos.${*}.txt solutions/backup.${*}.txt

revert_%:
	cp solutions/backup.${*}.txt ../assembler/InsAsmRepos/DefaultInsAsmRepos.${*}.txt

insert_%:
	cp solutions/${*}.txt ../assembler/InsAsmRepos/DefaultInsAsmRepos.${*}.txt

clean:
	-rm *.txt *.sass *.so *.cubin
	-rm tinygrad_gen
