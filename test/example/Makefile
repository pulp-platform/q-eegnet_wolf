PULP_APP = test

SRC_FOLDER = $(EEGNET_WOLF_SRC_PATH)

PULP_APP_CL_SRCS = \
	cluster.c \
	$(SRC_FOLDER)/cl/func/dot_prod.c \
	$(SRC_FOLDER)/cl/func/kernels/dot_prod_i8v.c

PULP_APP_FC_SRCS = test.c

PULP_LDFLAGS += -lplpdsp
PULP_CFLAGS = -O3 -g

include $(PULP_SDK_HOME)/install/rules/pulp_rt.mk
