INC  := -I.
WARN := -Wall -Wextra -Wpedantic
OPT  := -O3
FLGS := -fPIC

CFLAGS=$(INC) $(WARN) $(OPT) $(FLGS)

all:	libpolyfit.so examples/ex2

libpolyfit.so:	polyfit.c polyfit.h
	gcc $(CFLAGS) -shared -o libpolyfit.so polyfit.c

examples/ex2:	libpolyfit.so examples/ex2.c
	export LD_RUN_PATH=`pwd`; \
	gcc $(CFLAGS) -o $@ examples/ex2.c -L`pwd` -lpolyfit -lm

clean:
	/bin/rm -f libpolyfit.so examples/ex2
	find . -type f -name '*.py[co]' -print0 | xargs -0 -n 25 rm -f || true
	find . -type d -name '__pycache__' -print0 | xargs -0 -n 25 rm -rf || true
