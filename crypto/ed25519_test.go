package crypto


import (
	"bytes"
	"crypto/rand"
	"testing"

	"github.com/stretchr/testify/assert"
	"golang.org/x/crypto/ed25519"
)
// test Ed25519 benchmark
func TestEd25519Signer_Sign(t *testing.T) {
	signer := &ed25519Signer{}

	pubKey, privKey, err := ed25519.GenerateKey(rand.Reader)
	assert.Nil(t, err)
	assert.NotNil(t, pubKey)
	assert.NotNil(t, privKey)

	msg := bytes.NewBufferString("this is a string for ed25519 signer test").Bytes()

	// sign
	sig, err := signer.Sign(privKey, msg, nil)
	assert.Nil(t, err)
	assert.NotEmpty(t, sig)

	// verify ok
	ok, err := signer.Verify(pubKey, sig, msg, nil)
	assert.Nil(t, err)
	assert.Equal(t, true, ok)

	// verify notok
	notok, err := signer.Verify(pubKey, sig[1:], msg, nil)
	assert.Nil(t, err)
	assert.Equal(t, false, notok)
}

func TestEd25519Signer_Verify(t *testing.T) {
	signer := &ed25519Signer{}

	pubKey, privKey, err := ed25519.GenerateKey(rand.Reader)
	assert.Nil(t, err)
	assert.NotNil(t, pubKey)
	assert.NotNil(t, privKey)

	msg := bytes.NewBufferString("this is a string for ed25519 signer test").Bytes()

	// sign
	sig, err := signer.Sign(privKey, msg, nil)
	assert.Nil(t, err)
	assert.NotEmpty(t, sig)

	// verify ok
	ok, err := signer.Verify(pubKey, sig, msg, nil)
	assert.Nil(t, err)
	assert.Equal(t, true, ok)

	// verify notok
	notok, err := signer.Verify(pubKey, sig, bytes.NewBufferString("this is another string").Bytes(), nil)
	assert.Nil(t, err)
	assert.Equal(t, false, notok)
}

func BenchmarkEd25519Signer_Sign(b *testing.B) {
	signer := &ed25519Signer{}

	_, privKey, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		b.FailNow()
	}

	msg := bytes.NewBufferString("this is a string for ed25519 signer benchmark").Bytes()
	for i := 0; i <= b.N; i++ {
		signer.Sign(privKey, msg, nil)
	}
}

func BenchmarkEd25519Signer_Verify(b *testing.B) {
	signer := &ed25519Signer{}

	pubKey, privKey, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		b.FailNow()
	}

	msg := bytes.NewBufferString("this is a string for ed25519 signer benchmark").Bytes()

	// sign
	sig, err := signer.Sign(privKey, msg, nil)
	if err != nil {
		b.FailNow()
	}

	for i := 0; i <= b.N; i++ {
		signer.Verify(pubKey, sig, msg, nil)
	}
}