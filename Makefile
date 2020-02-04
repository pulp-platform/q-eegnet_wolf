PULP_APP = eegnet

PULP_APP_FC_SRCS = \
    src/fc/main.c

PULP_APP_CL_SRCS = \
    src/cl/cluster.c \
	src/cl/input.c \
	src/cl/net/model.c \
	src/cl/net/layer1.c \
	src/cl/net/layer2.c \
	src/cl/net/layer3.c \
	src/cl/net/layer4.c \
	src/cl/net/layer5.c \
	src/cl/net/net.c \
	src/cl/func/conv.c \
	src/cl/func/xcorr.c \
	src/cl/func/transform.c \
	src/cl/func/flip.c \
	src/cl/func/dotp.c \

PULP_CFLAGS = -O3 -g -DROUND

# flip layer 1 and layer 3 for faster dot product implementatoin
PULP_CFLAGS += "-DFLIP_LAYERS"

# use parallel processing
PULP_CFLAGS += "-DPARALLEL"

# scale data inside the convolution
PULP_CFLAGS += "-DINTRINSIC_SCALE"

# Access data via dma streaming
PULP_CFLAGS += "-DDMA_STREAM"

# Use Cross Correlation instead of Convolution
PULP_CFLAGS += "-DCROSS_CORRELATE"

# convolution version used
PULP_CFLAGS += "-DCONV_VERSION=2"

PULP_LDFLAGS += -lplpdsp

include $(PULP_SDK_HOME)/install/rules/pulp_rt.mk
