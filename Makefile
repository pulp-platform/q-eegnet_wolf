PULP_APP = eegnet

PULP_APP_FC_SRCS = \
    src/fc/main.c

PULP_APP_CL_SRCS = \
    src/cl/cluster.c \
	src/cl/net/layer1.c \
	src/cl/func/conv.c \
	src/cl/func/transform.c

PULP_CFLAGS = -O3 -g -DROUND

# use parallel processing
#PULP_CFLAGS += "-DPARALLEL"

PULP_LDFLAGS += -lplpdsp

include $(PULP_SDK_HOME)/install/rules/pulp_rt.mk
