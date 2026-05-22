# Convenience top-level Makefile for siamize. Wraps the CMake build and adds
# code-formatting / cleanup targets. CMake is still the primary build system
# (this Makefile just shells out to it).
#
# Targets:
#   make            — configure + build (Release) under build/
#   make clean      — remove build/
#   make pretty     — format all source (astyle on C++, black on Python)
#   make pretty-cpp — astyle on src/*.{cpp,h}
#   make pretty-py  — black on py/ and tools/
#   make test       — run tests/run_regression.sh (needs models/ populated)

BUILD_DIR ?= build
BUILD_TYPE ?= Release

.PHONY: all build clean pretty pretty-cpp pretty-py test

all: build

build:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)
	cmake --build $(BUILD_DIR) --config $(BUILD_TYPE) --parallel

clean:
	rm -rf $(BUILD_DIR)

pretty: pretty-cpp pretty-py

# astyle settings borrowed from MCX (https://github.com/fangq/mcx),
# itself derived from https://github.com/nlohmann/json.
pretty-cpp:
	astyle \
	    --style=attach \
	    --indent=spaces=4 \
	    --indent-modifiers \
	    --indent-switches \
	    --indent-preproc-block \
	    --indent-preproc-define \
	    --indent-col1-comments \
	    --pad-oper \
	    --pad-header \
	    --align-pointer=type \
	    --align-reference=type \
	    --add-brackets \
	    --convert-tabs \
	    --close-templates \
	    --lineend=linux \
	    --preserve-date \
	    --suffix=none \
	    --formatted \
	    --break-blocks \
	    "src/*.cpp" "src/*.h"

# Python formatting via black (PEP 8 conformant, default 88-col line length).
pretty-py:
	black py/ tools/

test:
	tests/run_regression.sh
