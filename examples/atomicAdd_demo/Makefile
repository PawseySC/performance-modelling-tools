

NVCC ?= nvcc
CFLAGS ?= 

BLOCKSIZE ?= 64
SIZE ?= 64

comparesums.exe:
	$(NVCC) $(CFLAGS) -DBLOCK_X=$(BLOCKSIZE) -DSIZE=$(SIZE) sum.cu -o comparesums.exe

clean:
	rm ./comparesums.exe


.PHONY: clean

