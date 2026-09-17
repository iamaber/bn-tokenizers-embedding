package native

import "encoding/binary"

// EncodeBinary accepts repeated uint32 byte lengths followed by UTF-8 text.
// The response starts with a status byte: 1 followed by an error message, or 0
// followed by repeated uint32 ID counts and uint32 IDs. Integers are little-endian.
// JSON remains the control interface for loading, decoding and releasing models.
func (r *Registry) EncodeBinary(handle uint64, data []byte) []byte {
	var texts []string
	for len(data) > 0 {
		if len(data) < 4 {
			return encodeError("truncated text length")
		}
		n := uint64(binary.LittleEndian.Uint32(data))
		data = data[4:]
		if n > uint64(len(data)) {
			return encodeError("truncated text")
		}
		texts = append(texts, string(data[:int(n)]))
		data = data[int(n):]
	}
	batch, err := r.encodeBatch(handle, texts)
	if err != nil {
		return encodeError(err.Error())
	}
	size := 1
	for _, ids := range batch {
		size += 4 + 4*len(ids)
	}
	output := make([]byte, 1, size)
	for _, ids := range batch {
		output = binary.LittleEndian.AppendUint32(output, uint32(len(ids)))
		for _, id := range ids {
			output = binary.LittleEndian.AppendUint32(output, uint32(id))
		}
	}
	return output
}

func encodeError(message string) []byte { return append([]byte{1}, message...) }
