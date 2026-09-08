// Package main exposes the Go tokenizer through an owned-string C ABI.
package main

/*
#include <stdlib.h>
*/
import "C"

import (
	"github.com/iamaber/bn-tokenizers-embedding/internal/native"
	"unsafe"
)

var registry native.Registry

//export bntok_call
func bntok_call(request *C.char) *C.char {
	return C.CString(string(registry.Call([]byte(C.GoString(request)))))
}

//export bntok_free
func bntok_free(response *C.char) {
	C.free(unsafe.Pointer(response))
}

func main() {}
