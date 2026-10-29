CXX ?= c++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wpedantic
all: build/pocket
build/pocket: src/main.cpp
	mkdir -p build
	$(CXX) $(CXXFLAGS) src/main.cpp -o build/pocket
clean:
	rm -rf build
.PHONY: all clean
