# Agentic Demo Examples
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