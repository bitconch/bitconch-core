file(GLOB UNIT_TESTS "*.cpp")
add_executable( chainbase_test ${UNIT_TESTS}  )
target_link_libraries( chainbase_test  chainbase ${Boost_LIBRARIES} ${OPENSSL_LIBRARIES} ${PLATFORM_SPECIFIC_LIBS} )

