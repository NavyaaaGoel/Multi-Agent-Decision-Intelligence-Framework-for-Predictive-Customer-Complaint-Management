export PYTHONPATH := src:tests
PY ?= python3

all: generate train simulate        ## full reproducible run (~4 min)
generate: ; $(PY) -m cdi generate
train:    ; $(PY) -m cdi train
simulate: ; $(PY) -m cdi simulate
demo:     ; $(PY) -m cdi demo
test:     ; $(PY) -m unittest discover -s tests -v
serve:    ; $(PY) -m cdi serve --port 8000
