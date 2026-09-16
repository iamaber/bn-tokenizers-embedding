// Package main exposes the Go tokenizer through owned C buffers.
package main

/*
#include <stdlib.h>
*/
import "C"

import (
	"math"
	"unsafe"

	"github.com/iamaber/bn-tokenizers-embedding/internal/native"
)

var registry native.Registry

//export bntok_encode
func bntok_encode(handle C.ulonglong, data *C.char, length C.size_t, outputLength *C.size_t) *C.char {
	if outputLength == nil {
		return nil
	}
	var response []byte
	if uint64(length) > math.MaxInt32 || (data == nil && length != 0) {
		response = append([]byte{1}, "invalid encode input size"...)
	} else {
		response = registry.EncodeBinary(uint64(handle), C.GoBytes(unsafe.Pointer(data), C.int(length)))
	}
	*outputLength = C.size_t(len(response))
	return (*C.char)(C.CBytes(response))
}

//export bntok_call
func bntok_call(request *C.char) *C.char {
	return C.CString(string(registry.Call([]byte(C.GoString(request)))))
}

//export bntok_free
func bntok_free(response *C.char) {
	C.free(unsafe.Pointer(response))
}

func main() {}
