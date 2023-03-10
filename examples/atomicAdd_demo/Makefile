

CC ?= hipcc
CFLAGS ?= 

BLOCKSIZE ?= 64
SIZE ?= 64

comparesums.exe:
	$(CC) $(CFLAGS) -DBLOCK_X=$(BLOCKSIZE) -DSIZE=$(SIZE) sum.cpp -o comparesums.exe

clean:
	rm ./comparesums.exe


.PHONY: clean

