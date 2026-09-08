GO ?= go

.PHONY: build test fetch prepare train evaluate matrix
build:
	$(GO) build -o bin/bntok ./cmd/bntok

test:
	$(GO) test -race ./...
	$(GO) vet ./...

fetch: build
	bin/bntok fetch

prepare: build
	bin/bntok prepare
	bin/bntok audit

train: build
	bin/bntok train

evaluate: build
	bin/bntok evaluate

matrix: build
	mkdir -p reports
	set -e; for size in 16000 32000 48000; do \
	  for algorithm in unigram bpe; do \
	    bin/bntok train -algorithm $$algorithm -vocab $$size -out models/$$algorithm-$$size.json; \
	    bin/bntok evaluate -model models/$$algorithm-$$size.json > reports/$$algorithm-$$size-validation.json; \
	  done; \
	done
