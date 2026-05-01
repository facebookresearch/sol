# PROJECT: fcontext
cmake_minimum_required(VERSION 3.10)
project(fcontext C)

option(BUILD_TEST "build test app" 0)

if (NOT CMAKE_MODULE_PATH)
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
endif()

if (MSVC AND NOT CMAKE_C_COMPILER_ID MATCHES "Clang")
    enable_language(CXX ASM_MASM)
else()
    enable_language(CXX ASM)
endif()

if(MSVC)
  add_definitions(-D_ITERATOR_DEBUG_LEVEL=0)
  add_definitions(-D_HAS_EXCEPTIONS=0)
endif()
add_definitions(-DBOOST_CONTEXT_EXPORT=)

set(HEADER "include/fcontext/fcontext.h")
set(SOURCES "source/stack.c")

# OS
if (APPLE)
    set(CPU_ARCH "combined")
    set(ASM_EXT "all_macho_gas.S")
elseif (ANDROID)
    # Android
    if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm")
        set(CPU_ARCH "arm")
        set(ASM_EXT "aapcs_elf_gas.S")
    elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
        set(CPU_ARCH "arm64")
        set(ASM_EXT "aapcs_elf_gas.S")
    elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "i686")
        set(CPU_ARCH "i386")
        set(ASM_EXT "sysv_elf_gas.S")
    elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64")
        set(CPU_ARCH "x86_64")
        set(ASM_EXT "sysv_elf_gas.S")
    endif()
elseif (UNIX)
    # PC (ARM)
    if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm")
        set(CPU_ARCH "arm")
        set(ASM_EXT "aapcs_elf_gas.S")
    elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
        set(CPU_ARCH "arm64")
        set(ASM_EXT "aapcs_elf_gas.S")
    # PC (x86/x64)
    elseif (CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(CPU_ARCH "x86_64")
        set(ASM_EXT "sysv_elf_gas.S")
    else()
        set(CPU_ARCH "i386")
        set(ASM_EXT "sysv_elf_gas.S")
    endif()
elseif (WIN32)
    # Windows PC
    if (CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(CPU_ARCH "x86_64")
    else()
        set(CPU_ARCH "i386")
    endif()

    if (CMAKE_C_COMPILER_ID MATCHES "Clang")
        set(ASM_EXT "ms_pe_clang_gas.S")
    elseif (MSVC)
        set(ASM_EXT "ms_pe_masm.asm")
    elseif (CMAKE_C_COMPILER_ID MATCHES "GNU")
        set(ASM_EXT "ms_pe_gas.S")
    endif()
    
endif()

set(ASM_SOURCES "asm/make_${CPU_ARCH}_${ASM_EXT}"
                "asm/jump_${CPU_ARCH}_${ASM_EXT}"
                "asm/ontop_${CPU_ARCH}_${ASM_EXT}")

add_library(fcontext STATIC ${SOURCES} ${ASM_SOURCES})
target_include_directories(fcontext
    PRIVATE include/fcontext
    INTERFACE include)

set_target_properties(fcontext PROPERTIES FOLDER Deps ${IOS_GENERAL_PROPERTIES})

install(TARGETS fcontext DESTINATION lib)
install(FILES ${HEADER} DESTINATION include/fcontext)

if (BUILD_TEST)
    add_executable(test_fcontext test/test_fcontext.c)
    target_link_libraries(test_fcontext fcontext)
endif()

