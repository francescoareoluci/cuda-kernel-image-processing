#
#  Make file for cuda_image_filering_global
#

#
# Macros
#
IMG_LDFLAG	= -lpng
LDFLAGS 	= $(IMG_LDFLAG) -lm

CUDA_INCFLAG	= -I/home/francesco/NVIDIA_CUDA-10.1_Samples/common/inc
INCFLAGS	= $(CUDA_INCFLAG)

CC		= nvcc
CFLAGS		= -gencode arch=compute_50,code=sm_50
		  --fmad=false \
		  -O3 -std=c++11

CPP_SRCS	= kernel.cpp \
		  image.cpp 

CPP_HDRS	= kernel.h \
		  image.h \
		  gpu_convolution.h 

CU_SRCS		= main.cu \
		  gpu_convolution.cu

CU_HDRS		= 

CPP_OBJS	= $(CPP_SRCS:.cpp=.o) 
CU_OBJS		= $(CU_SRCS:.cu=.o)
TARGET		= kernel_convolution

CPP_DEPS	= $(CPP_SRCS:.cpp=.d)
CU_DEPS		= $(CU_SRCS:.cu=.d)
DEP_FILE	= Makefile.dep

#
# Suffix rules
#
.SUFFIXES: .cpp
.cpp.o:
	$(CC) $(INCFLAGS) $(CFLAGS)  -c $<

.SUFFIXES: .cu
.cu.o:
	$(CC) $(INCFLAGS) $(CFLAGS)  -c $<

.SUFFIXES: .d
.cpp.d:
	$(CC) $(INCFLAGS) -M $< > $*.d
.cu.d:
	$(CC) $(INCFLAGS) -M $< > $*.d

#
# Generating the target
#
all: $(DEP_FILE) $(TARGET) 

#
# Linking the execution file
#
$(TARGET) : $(CU_OBJS) $(CPP_OBJS) 
	$(CC) -o $@ $(CU_OBJS) $(CPP_OBJS) $(LDFLAGS)

#
# Generating and including dependencies
#
depend: $(DEP_FILE)
$(DEP_FILE) : $(CPP_DEPS) $(CU_DEPS)
	cat $(CPP_DEPS) $(CU_DEPS) > $(DEP_FILE)
ifeq ($(wildcard $(DEP_FILE)),$(DEP_FILE))
include $(DEP_FILE)
endif

#
# Cleaning the files
#
clean:
	rm -f $(CU_OBJS) $(CPP_OBJS) $(CPP_DEPS) $(CU_DEPS) $(DEP_FILE) $(TARGET) *~
