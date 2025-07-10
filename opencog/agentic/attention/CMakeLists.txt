# Attention Allocation Engine library
SET(AGENTIC_ATTENTION_SOURCES
    AttentionAllocationEngine.cc
)

ADD_LIBRARY(agentic-attention SHARED ${AGENTIC_ATTENTION_SOURCES})

# Include GGML headers
TARGET_INCLUDE_DIRECTORIES(agentic-attention PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

TARGET_LINK_LIBRARIES(agentic-attention
    cogutil
    ${Boost_LIBRARIES}
    agentic-kernel
)

INSTALL(TARGETS agentic-attention
    LIBRARY DESTINATION lib/opencog
    ARCHIVE DESTINATION lib/opencog
)

INSTALL(FILES 
    AttentionAllocationEngine.h
    DESTINATION include/opencog/agentic/attention
)