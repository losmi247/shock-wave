OPENCV_FLAGS=`pkg-config --cflags --libs opencv4`

build-dir:
	mkdir -p build

all: flow.cc build-dir
	g++ -o ./build/flow flow.cc ${OPENCV_FLAGS} -l pthread

clean:
	rm -rf ./build/flow