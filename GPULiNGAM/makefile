# Compiler settings
CC=nvcc
CFLAGS=-std=c++11 -arch=sm_60 -I.

# Source and object files
SOURCES=$(wildcard *.cu)
OBJECTS=$(SOURCES:.cu=.o)

# Target executable
TARGET=GPULiNGAM

# Default target
all: $(TARGET)

# Linking the executable
$(TARGET): $(OBJECTS)
	$(CC) -o $@ $^

# Compiling source files into object files
%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up
clean:
	rm -f $(TARGET) $(OBJECTS)
