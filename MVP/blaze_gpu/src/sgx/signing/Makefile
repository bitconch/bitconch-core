SGX_SDK ?= /opt/sgxsdk
SIGN_ENCLAVE ?= 1
PRIV_KEY ?= ../keys/private_key.pem
PUB_KEY ?= ../keys/public_key.pem
LIBS_PATH ?= ../../../libs
OUT ?= ../../../dist

SGX_COMMON_CFLAGS := -m64 -O2
SGX_LIBRARY_PATH := $(SGX_SDK)/lib64
SGX_ENCLAVE_SIGNER := $(SGX_SDK)/bin/x64/sgx_sign
SGX_EDGER8R := $(SGX_SDK)/bin/x64/sgx_edger8r

Trusted_C_Flags := -Wno-implicit-function-declaration -std=c11 $(SGX_COMMON_CFLAGS) -nostdinc  -fpie -fstack-protector \
	-IInclude -I. -I$(SGX_SDK)/include -I$(SGX_SDK)/include/tlibc -I$(SGX_SDK)/include/libcxx -fno-builtin-printf \
	-I. -I../../sgx-ecc-ed25519 -fvisibility=hidden

Untrusted_C_Flags := -fPIC -O0 -g -Wno-attributes -I$(SGX_SDK)/include -I. -I../../sgx-ecc-ed25519 
Test_C_Flags := $(Untrusted_C_Flags)

Link_Flags := $(SGX_COMMON_CFLAGS) -Wl,--no-undefined -nostdlib -nodefaultlibs -nostartfiles -L$(SGX_LIBRARY_PATH) \
	-Wl,--whole-archive -lsgx_trts -Wl,--no-whole-archive \
	-L$(LIBS_PATH) -led25519.sgx.static \
	-Wl,--start-group -lsgx_tstdc -lsgx_tcxx -lsgx_tkey_exchange -lsgx_tcrypto -lsgx_tservice -Wl,--end-group \
	-Wl,-Bstatic -Wl,-Bsymbolic -Wl,--no-undefined \
	-Wl,-pie,-eenclave_entry -Wl,--export-dynamic  \
	-Wl,--defsym,__ImageBase=0 \
	-Wl,--version-script=./signing.lds

Trusted_C_Files := $(filter $(wildcard *_trusted.c), $(wildcard *.c))
Trusted_C_Files += signing_t.c
Trusted_C_Objects := $(Trusted_C_Files:.c=.o)

Untrusted_C_Files := $(filter $(wildcard *_untrusted.c), $(wildcard *.c))
Untrusted_C_Files += signing_u.c
Untrusted_C_Objects := $(Untrusted_C_Files:.c=.o)

.PHONY: all run

ifneq ($(SIGN_ENCLAVE), 0)
all: signing_u.c signing_t.c signing.signed.so libsigning.so
else
all: signing_u.c signing_t.c signing.so libsigning.so
	@echo "Build enclave signing.so success!"
	@echo
	@echo "**********************************************************************************************"
	@echo "PLEASE NOTE: In this mode, please sign the enclave first using Two Step Sign mechanism, before"
	@echo "you run the app to launch and access the enclave."
	@echo "**********************************************************************************************"
	@echo
endif

run: all

signing_t.c: $(SGX_EDGER8R) signing.edl
	@echo "GEN  =>  $@"
	@$(SGX_EDGER8R) --trusted signing.edl --search-path $(SGX_SDK)/include

signing_u.c: $(SGX_EDGER8R) signing.edl
	@echo "GEN  =>  $@"
	@$(SGX_EDGER8R) --untrusted signing.edl --search-path $(SGX_SDK)/include

$(Trusted_C_Objects): %.o: %.c
	@echo "CC  <=  $<"
	$(CC) $(Trusted_C_Flags) -c $< -o $@

$(Untrusted_C_Objects): %.o: %.c
	@echo "CC  <=  $<"
	$(CC) $(Untrusted_C_Flags) -c $< -o $@

signing.so: signing_t.o $(Trusted_C_Objects)
	@echo "LINK =>  $@"
	$(CC) $^ -o $@ $(Link_Flags)
	mkdir -p $(OUT)
	cp $@ $(OUT)

signing.signed.so: signing.so
	@echo "SIGN =>  $@"
	$(SGX_ENCLAVE_SIGNER) gendata -enclave $< -config signing.config.xml -out /tmp/enclave_hash.hex
	openssl dgst -sha256 -out /tmp/signature.hex -sign $(PRIV_KEY) -keyform PEM /tmp/enclave_hash.hex
	$(SGX_ENCLAVE_SIGNER) catsig -enclave $< -config signing.config.xml -out $@ -key $(PUB_KEY) -sig /tmp/signature.hex -unsigned /tmp/enclave_hash.hex
	mkdir -p $(OUT)
	cp $@ $(OUT)

libsigning.so: signing_u.o signing_untrusted.o $(Untrusted_C_Objects)
	@echo "LINK =>  $@"
	$(CC) $^ -o $@ -shared -L$(SGX_LIBRARY_PATH) -lsgx_uae_service -lsgx_ukey_exchange -lsgx_urts -L$(LIBS_PATH) -led25519.static
	mkdir -p $(OUT)
	cp $@ $(OUT)
	cp signing_public.h $(OUT)

clean:
	@rm -f signing_t.* signing_u.* $(Trusted_C_Objects) $(Untrusted_C_Objects) signing.signed.so signing.so libsigning.so
