# Agentic Kernel base library
SET(AGENTIC_KERNEL_SOURCES
    AgenticKernel.cc
)

ADD_LIBRARY(agentic-kernel SHARED ${AGENTIC_KERNEL_SOURCES})

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
    DESTINATION include/opencog/agentic/kernel
)