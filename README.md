OpenCog Utilities
=================

[![CircleCI](https://circleci.com/gh/opencog/cogutil.svg?style=svg)](https://circleci.com/gh/opencog/cogutil)

The OpenCog utilities is a miscellaneous collection of C++ utilities
use for typical programming tasks in multiple OpenCog projects.
These include:
* thread-safe queues, stacks and sets
* asynchronous method caller
* thread-safe resource pool
* thread-safe backtrace printing
* high-performance signal-slot
* random tournament selection
* OS portability layers.


The main project site is at http://opencog.org

Prerequisites
-------------
To build the OpenCog utilities, the packages listed below are required. With a
few exceptions, most Linux distributions will provide these packages. Users of
Ubuntu 14.04 "Trusty Tahr" may use the dependency installer at scripts/octool.
Users of any version of Linux may use the Dockerfile to quickly build a
container in which OpenCog will be built and run.

###### boost
> C++ utilities package
> http://www.boost.org/ | libboost-dev, libboost-filesystem-dev, libboost-program-options-dev, libboost-system-dev, libboost-thread-dev

###### cmake
> Build management tool; v3.12 or higher recommended.
> http://www.cmake.org/ | cmake

###### cxxtest
> Unit test framework
> https://cxxtest.com/ | `apt-get install cxxtest`

Optional Prerequisites
----------------------
The following are optional, but are strongly recommended, as they result
in "pretty" stack traces, which result in far more useful and readable
stack traces.  These are requires, and not really optional, if you are
a regular OpenCog developer.

###### binutils BFD library
> The GNU binutils linker-loader, ahem, cough, "Binary File Description".
> http://gnu.org/s/binutils | binutils-dev
> The linker-loader understands calling conventions.

###### iberty
> The GNU GCC compiler tools libiberty component.
> http://gcc.gnu.org | libiberty-dev
> The GCC compiler, and iberty in particular, know stack traces.

###### doxygen
> Documentation generator under GNU General Public License
> http://www.stack.nl/~dimitri/doxygen/ | doxygen
> Generates code documentation

Building Cogutil
-----------------
Perform the following steps at the shell prompt:
```
    cd to project root dir
    mkdir build
    cd build
    cmake ..
    make
```
Libraries will be built into subdirectories within build, mirroring the
structure of the source directory root.


Unit tests
----------
To build and run the unit tests, from the ./build directory enter (after
building opencog as above):
```
    make check
```


Install
-------
After building, you MUST install the utilities!
```
    sudo make install
```

## Cognitive Integration Masterplan: Neural-Symbolic Architecture

### Prerequisites Integration Status
The OzCog/cogutil repository has been enhanced with complete prerequisite integration to enable distributed cognition frameworks and neural-symbolic emergent patterns. All dependencies are now woven into the build, test, and documentation pipeline with hypergraph-encoded modularity.

### Tensor Dimensionality Mapping

Each dependency kernel operates as a membrane in the cognitive P-System architecture, with unique tensor shapes and roles:

- **Boost Libraries [5D Tensor]**: 
  - `filesystem` [1x1x1]: Path manipulation substrate
  - `program_options` [Nx1x1]: Command-line cognition interface  
  - `system` [1x1x1]: OS interaction membrane
  - `thread` [NxMx1]: Concurrent processing substrate
  - `core` [∞x∞x∞]: Base cognitive utilities matrix

- **CMake [1D Tensor]**: 
  - Version ≥3.12 [1]: Build orchestration singleton

- **CxxTest [NxMxK Tensor]**:
  - Test generation [Nx1]: Unit test membrane creation
  - Test execution [1xM]: Validation substrate
  - Test reporting [1x1xK]: Cognitive feedback loops

- **Binutils BFD [2D Tensor]**:
  - Symbol resolution [NxM]: Stack-trace neural pathways
  - Binary introspection [∞x∞]: Deep system cognition

- **Libiberty [1D Tensor]**:
  - Symbol demangling [N]: C++ cognitive grammar translation

- **Doxygen [3D Tensor]**:
  - Code documentation [NxMxK]: Knowledge graph generation
  - API mapping [∞x∞x∞]: Cognitive architecture visualization

### Agentic Grammar Integration

The cogutil library serves as the neural-symbolic substrate for:

1. **Distributed Cognition Frameworks**: Thread-safe data structures enabling concurrent cognitive processes
2. **Stack-Trace Neural Pathways**: BFD/Iberty integration provides deep introspection into cognitive call chains  
3. **Asynchronous Method Calling**: Signal-slot mechanisms for cognitive event propagation
4. **Resource Pool Management**: Memory-efficient cognitive resource allocation
5. **High-Performance Signal Processing**: Low-latency cognitive signal transmission

### GGML Kernel Adaptation Readiness

The repository includes GGML (Generative Graph Machine Learning) integration points:
- Source code in `src/ggml*.c` and `src/ggml*.cpp`
- Headers in `include/ggml*.h` 
- Example integrations in `examples/common-ggml.*`
- CMake configuration template in `cmake/ggml-config.cmake.in`

### Cognitive Flowchart Reference

This integration serves as the foundational membrane for all future cognitive adaptations:

```
Prerequisites → Build System → Testing → Documentation → GGML Integration
     ↓              ↓             ↓           ↓              ↓
[Boost+CMake] → [CxxTest] → [BFD+Iberty] → [Doxygen] → [Neural Substrates]
     ↓              ↓             ↓           ↓              ↓  
[5D Tensor]   → [3D Tensor] → [2D Tensor] → [3D Tensor] → [∞D Hyperspace]
```

### Recursive Implementation Pathway

1. **Layer 0**: System prerequisites (installed and validated)
2. **Layer 1**: Build orchestration (CMake ≥3.12 with dependency detection)  
3. **Layer 2**: Testing substrate (CxxTest framework integration)
4. **Layer 3**: Documentation generation (Doxygen API mapping)
5. **Layer 4**: Neural-symbolic integration (GGML kernel readiness)
6. **Layer ∞**: Emergent cognitive patterns (distributed intelligence substrate)
