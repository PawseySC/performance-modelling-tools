
CXX = hipcc
buildtype = hip
DEFS = -DUSEHIP 
CXXFLAGS = -std=c++17 -Xcompiler -O2 \
-I$(shell pwd)/../profile_util/include/ -Iinclude/ \
$(DEFS)
LDFLAGS = -L$(shell pwd)/../profile_util/lib/ -lprofile_util_$(buildtype) -Wl,-rpath=$(shell pwd)/../profile_util/lib/

# Target executable name:
EXE = bin/warm_up_test.$(buildtype).exe

OBJS = obj/main.o \
obj/WarmupGPU.o \
obj/logger.o

.PHONY: libprofile_util $(EXE)

all : libprofile_util $(EXE) 

BLUE='\034[0;31m'
NC='\033[0m' # No Color

libprofile_util:
	@echo "Making profile_util library" ;
	$(MAKE) -C ../profile_util/ COMPILER=$(CXX) EXTRAFLAGS=$(DEFS); 
	cp ../profile_util/lib/libprofile_util.so ../profile_util/lib/libprofile_util_$(buildtype).so 

$(EXE) : $(OBJS)
	@echo "Making $(EXE) " ;
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -v $(OBJS) -o $@ -Iinclude

obj/%.o: src/%.cpp include/*.h 
	$(CXX) $(CXXFLAGS) -Iinclude/ -c $< -o $@

obj/%.o: src/%.cu include/*.h 
	$(CXX) $(CXXFLAGS) -Iinclude/ -c $< -o $@

clean:
	rm $(OBJS) $(EXE)
	$(MAKE) -C profile_util/ clean  
	
