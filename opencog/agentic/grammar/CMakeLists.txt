# Grammar processing kernels
SET(AGENTIC_GRAMMAR_SOURCES
    TokenParsingAgent.cc
)

ADD_LIBRARY(agentic-grammar SHARED ${AGENTIC_GRAMMAR_SOURCES})

TARGET_LINK_LIBRARIES(agentic-grammar
    agentic-kernel
    cogutil
    ${Boost_LIBRARIES}
)

INSTALL(TARGETS agentic-grammar
    LIBRARY DESTINATION lib/opencog
    ARCHIVE DESTINATION lib/opencog
)

INSTALL(FILES 
    TokenParsingAgent.h
    DESTINATION include/opencog/agentic/grammar
)