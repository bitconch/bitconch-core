# will be implictly used for any compilation unit if not overrided by SYSTEM_INCLUDE_FOLDERS parameter
# these directories go as -isystem <dir> to avoid warnings from code of third-party libraries

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/bccio.bios/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bccio.bios/)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/bccio.msig/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bccio.msig/)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/bccio.system/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bccio.system/)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/bccio.token/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bccio.token/)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/bccio.wrap/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bccio.wrap/)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/old_versions/v1.6.0-rc3/bccio.bios/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/old_versions/v1.6.0-rc3/bccio.bios/)
