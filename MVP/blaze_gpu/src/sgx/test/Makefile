SGX_SDK ?= /opt/sgxsdk
LIBS_PATH ?= ../../../libs
OUT ?= ../../../dist

C_Flags := -O2 -fpic -I. -I$(SGX_SDK)/include -I$(OUT) -I../../sgx-ecc-ed25519

C_Files := $(wildcard *.c)
C_Objects := $(C_Files:%.c=%.o)

.PHONY: all run
all: $(OUT)/signing_test
run: all

%.o: %.c
	@echo "CC  <=  $<"
	$(CC) $(C_Flags) -c $< -o $@

$(OUT)/signing_test: $(C_Objects)
	@mkdir -p $(OUT)
	$(CC) $^ -o $@ -L$(OUT) -L$(LIBS_PATH) -lsigning -led25519.static

clean:
	@rm -rf $(C_Objects) $(OUT)/signing_test
