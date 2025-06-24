obj-m += tdma.o

KERN_DIR ?= /lib/modules/$(shell uname -r)/build/

modules:
	make -C $(KERN_DIR) M=$(PWD) modules

clean:
	make -C $(KERN_DIR) M=$(PWD) clean
