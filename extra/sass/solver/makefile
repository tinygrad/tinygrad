ARCH=sm_86
LIBS := libcublasLt.txt libcublas.txt
TINY_GEN_DIR=tiny_gen


find_%:
	find /usr/local/cuda/ -name "*${*}*"
	
stdlibs/%.so:
	mkdir -p stdlibs
	cp /usr/local/cuda/targets/x86_64-linux/lib/${*}.so $@

${TINY_GEN_DIR}/%_cu:
	mkdir -p ${TINY_GEN_DIR} && \
	export DUMP_CUDA_DIR=$$(pwd)/${TINY_GEN_DIR} && \
	export DUMP_CUDA_PREFIX=${*} && \
	cd ../../.. && \
	export PYTHONPATH="." && \
	NV=1 python test/${*}.py

${TINY_GEN_DIR}/%_txt:
	export dir=${TINY_GEN_DIR};\
	export PYTHONPATH="/home/alvy/gbin/tinygrad"; \
	for cufile in ${TINY_GEN_DIR}/${*}_*.cu; \
		do \
			echo $${cufile};\
			filename_ext=$$(basename -- $${cufile}); \
			base="$${filename_ext%.*}"; \
			nvcc --ptx -arch ${ARCH} -o $${dir}/$${base}.ptx $${cufile}; \
			ptxas -arch=${ARCH} -m64 -o $${dir}/$${base}.cubin $${dir}/$${base}.ptx; \
			python disasm.py -a ${ARCH} -o $${dir}/$${base}.sass $${dir}/$${base}.cubin; \
			python solve.py -a ${ARCH} -s $${dir}/$${base}.sass -d $${dir}/$${base}.txt; \
		done; \

tiny_%.txt:
	python merge.py ${*}.txt ${TINY_GEN_DIR}/*.txt

%.sass: cubins/%.cubin
	python disasm.py -a ${ARCH} ${<} -o ${@}

%.sass: stdlibs/%.so 
	python disasm.py -a ${ARCH} ${<} -o ${@}

stdlib_%.txt: %.sass
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
