PULP_APP = eegnet

PULP_APP_FC_SRCS = \
    src/fc/main.c

PULP_APP_CL_SRCS = \
    src/cl/cluster.c \
    src/cl/func/dot_prod.c \
    src/cl/func/kernels/dot_prod_i8v.c

PULP_CFLAGS = -O3 -g
PULP_LDFLAGS += -lplpdsp

include $(PULP_SDK_HOME)/install/rules/pulp_rt.mk
