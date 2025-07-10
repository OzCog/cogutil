# Agentic Kernel base library
SET(AGENTIC_KERNEL_SOURCES
    AgenticKernel.cc
    TensorShape.cc
    TensorMapper.cc
    Node.cc
    Link.cc
    ggml_stub.c
)

ADD_LIBRARY(agentic-kernel SHARED ${AGENTIC_KERNEL_SOURCES})

# Include GGML headers
TARGET_INCLUDE_DIRECTORIES(agentic-kernel PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

TARGET_LINK_LIBRARIES(agentic-kernel
    cogutil
    ${Boost_LIBRARIES}
)

INSTALL(TARGETS agentic-kernel
    LIBRARY DESTINATION lib/opencog
    ARCHIVE DESTINATION lib/opencog
)

INSTALL(FILES 
    AgenticKernel.h
    TensorShape.h
    TensorMapper.h
    Link.h
    Node.h
    DESTINATION include/opencog/agentic/kernel
)