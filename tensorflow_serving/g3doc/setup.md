# Installation

## Prerequisites

To compile and use TensorFlow Serving, you need to set up some prerequisites.

### Bazel

TensorFlow Serving requires Bazel 0.2.0 or higher. You can find the Bazel
installation instructions [here](http://bazel.io/docs/install.html).

If you have the prerequisites for Bazel, those instructions consist of the
following steps:

1.  Download the relevant binary from
    [here](https://github.com/bazelbuild/bazel/releases).
    Let's say you downloaded bazel-0.2.0-installer-linux-x86_64.sh. You would
    execute:

    ~~~shell
    cd ~/Downloads
    chmod +x bazel-0.2.0-installer-linux-x86_64.sh
    ./bazel-0.2.0-installer-linux-x86_64.sh --user
    ~~~
2.  Set up your environment. Put this in your ~/.bashrc.

    ~~~shell
    export PATH="$PATH:$HOME/bin"
    ~~~

### gRPC

Our tutorials use [gRPC](http://www.grpc.io) (0.13 or higher) as our RPC
framework. You can find the installation instructions
[here](https://github.com/grpc/grpc/tree/master/src/python/grpcio).

### Packages

To install TensorFlow Serving dependencies, execute the following:

~~~shell
sudo apt-get update && sudo apt-get install -y \
        build-essential \
        curl \
        git \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python-dev \
        python-numpy \
        python-pip \
        software-properties-common \
        swig \
        zip \
        zlib1g-dev
~~~

## Installing from source

### Clone the TensorFlow Serving repository

~~~shell
git clone --recurse-submodules https://github.com/tensorflow/serving
~~~

`--recurse-submodules` is required to fetch TensorFlow, gRPC, and other
libraries that TensorFlow Serving depends on. Note that these instructions
will install the latest master branch of TensorFlow Serving. If you want to
install a specific branch (such as a release branch), pass `-b <branchname>`
to the `git clone` command.

### Install prerequisites

Follow the Prerequisites section above to install all dependencies.
To configure TensorFlow, run

~~~shell
cd tensorflow
./configure
cd ..
~~~

Consult the [TensorFlow install instructions](
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md)
if you encounter any issues with setting up TensorFlow or its dependencies.


### Build

TensorFlow Serving uses Bazel to build. Use Bazel commands to build individual
targets or the entire source tree.

To build the entire tree, execute:

~~~shell
bazel build tensorflow_serving/...
~~~

Binaries are placed in the bazel-bin directory, and can be run using a command
like:

~~~shell
./bazel-bin/tensorflow_serving/example/mnist_inference
~~~

To test your installation, execute:

~~~shell
bazel test tensorflow_serving/...
~~~

See the [basic tutorial](serving_basic.md) and [advanced tutorial](serving_advanced.md)
for more in-depth examples of running TensorFlow Serving.
