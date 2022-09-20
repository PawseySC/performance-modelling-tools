
buildtype = omp
DEFS = -DUSEOPENMP
CXXFLAGS = -std=c++17 -O2 -fopenmp 

CXX = gcc
ifeq ($(CXX), nvcc)
CXXFLAGS = -Xcompiler -fPIC -Xcompiler -std=c++14 -O2
buildtype = cuda
DEFS = -DUSECUDA
endif
ifeq ($(CXX), hipcc)
buildtype = hip
DEFS = -DUSEHIP 
CXXFLAGS = -std=c++17 -O2
endif

CXXFLAGS += -I$(shell pwd)/../profile_util/include/ -Iinclude/ 
CXXFLAGS += $(DEFS) 
LDFLAGS = -L$(shell pwd)/../profile_util/lib/
ifeq ($(buildtype), hip)
LDFLAGS += -Wl,-rpath=$(shell pwd)/../profile_util/lib/
endif
ifeq ($(buildtype), omp)
LDFLAGS += -Wl,-rpath=$(shell pwd)/../profile_util/lib/
endif
LIBS = -lprofile_util_$(buildtype)

# Target executable name:
EXE = bin/warm_up_test.$(buildtype).exe

OBJS = obj/main.o \
obj/logger.o \
obj/WarmupGPU.o \

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
	$(CXX) $(CXXFLAGS) $(OBJS) $(LDFLAGS) -o $@ $(LIBS)

ifeq ($(buildtype), omp)
obj/WarmupGPU.o: src/WarmupGPU.cu include/*.h 
	cp src/WarmupGPU.cu src/WarmupGPU.cpp ;
	$(CXX) $(CXXFLAGS) -Iinclude/ -c src/WarmupGPU.cpp -o obj/WarmupGPU.o ;
	rm src/WarmupGPU.cpp ;
obj/%.o: src/%.cpp include/*.h 
	$(CXX) $(CXXFLAGS) -Iinclude/ -c $< -o $@
else 
obj/%.o: src/%.cpp include/*.h 
	$(CXX) $(CXXFLAGS) -Iinclude/ -c $< -o $@
obj/%.o: src/%.cu include/*.h 
	$(CXX) $(CXXFLAGS) -Iinclude/ -c $< -o $@
endif



clean:
	rm -f $(OBJS) $(EXE)
	$(MAKE) -C ../profile_util/ clean  
	
