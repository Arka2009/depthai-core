set(_internal_flags_sanitizer "-fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize=pointer-overflow")
set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} ${_internal_flags_sanitizer})
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${_internal_flags_sanitizer})
set(CMAKE_C_FLAGS_INIT ${CMAKE_C_FLAGS_INIT} ${_internal_flags_sanitizer})
set(CMAKE_CXX_FLAGS_INIT ${CMAKE_CXX_FLAGS_INIT} ${_internal_flags_sanitizer})
set(CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS} ${_internal_flags_sanitizer})
set(CMAKE_EXE_LINKER_FLAGS_INIT ${CMAKE_EXE_LINKER_FLAGS_INIT} ${_internal_flags_sanitizer})
set(CMAKE_SHARED_LINKER_FLAGS ${CMAKE_SHARED_LINKER_FLAGS} ${_internal_flags_sanitizer})
set(CMAKE_SHARED_LINKER_FLAGS_INIT ${CMAKE_SHARED_LINKER_FLAGS_INIT} ${_internal_flags_sanitizer})
set(CMAKE_MODULE_LINKER_FLAGS ${CMAKE_MODULE_LINKER_FLAGS} ${_internal_flags_sanitizer})
set(CMAKE_MODULE_LINKER_FLAGS_INIT ${CMAKE_MODULE_LINKER_FLAGS_INIT} ${_internal_flags_sanitizer})
set(_internal_flags_sanitizer)
