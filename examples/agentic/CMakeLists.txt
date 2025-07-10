# Agentic Demo Examples
SET(BASIC_DEMO_SOURCES
    basic_demo.cc
)

ADD_EXECUTABLE(basic_agentic_demo ${BASIC_DEMO_SOURCES})

TARGET_LINK_LIBRARIES(basic_agentic_demo
    agentic-grammar
    agentic-kernel
    cogutil
    ${Boost_LIBRARIES}
)

# Cognitive Kernel Genesis Demo
SET(COGNITIVE_DEMO_SOURCES
    cognitive_kernel_demo.cc
)

ADD_EXECUTABLE(cognitive_kernel_demo ${COGNITIVE_DEMO_SOURCES})

TARGET_LINK_LIBRARIES(cognitive_kernel_demo
    agentic-grammar
    agentic-kernel
    cogutil
    ${Boost_LIBRARIES}
)

# Install demo binaries
INSTALL(TARGETS basic_agentic_demo cognitive_kernel_demo
    RUNTIME DESTINATION bin
)

# Advanced demo with full attention system (optional)
IF(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    SET(SIMPLE_DEMO_SOURCES
        simple_agentic_demo.cc
    )

    ADD_EXECUTABLE(simple_agentic_demo ${SIMPLE_DEMO_SOURCES})

    TARGET_LINK_LIBRARIES(simple_agentic_demo
        agentic-grammar
        agentic-kernel
        cogutil
        ${Boost_LIBRARIES}
    )

    # Install demo binary
    INSTALL(TARGETS simple_agentic_demo
        RUNTIME DESTINATION bin
    )
ENDIF()