cmake_minimum_required(VERSION 3.16)
project(zipformer_rnnt_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Options
option(BUILD_RKNN "Build with RKNN NPU backend" ON)

# RKNN Runtime
if(BUILD_RKNN)
  # RKNN_API_PATH can point to either:
  #   - a split SDK dir with include/ and lib/ subdirs  (e.g. rknn-toolkit2/rknpu2/runtime/Linux/librknn_api)
  #   - a prefix like /usr/local where headers+lib are already installed
  set(RKNN_API_PATH "/usr/local" CACHE PATH "Path to RKNN API prefix (contains include/ and lib/)")
  include_directories(${RKNN_API_PATH}/include)
  link_directories(${RKNN_API_PATH}/lib)
  add_definitions(-DUSE_RKNN)
endif()

# sentencepiece (BPE tokenizer)
find_package(sentencepiece QUIET)
if(NOT sentencepiece_FOUND)
  message(WARNING "sentencepiece not found via find_package, trying pkg-config")
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(SENTENCEPIECE REQUIRED sentencepiece)
  include_directories(${SENTENCEPIECE_INCLUDE_DIRS})
  link_directories(${SENTENCEPIECE_LIBRARY_DIRS})
endif()

# libsndfile for WAV I/O
find_package(PkgConfig REQUIRED)
pkg_check_modules(SNDFILE REQUIRED sndfile)

include_directories(src)

set(SOURCES
  src/main.cpp
  src/audio_utils.cpp
  src/bpe_tokenizer.cpp
  src/zipformer_encoder.cpp
  src/rnnt_decoder.cpp
  src/rnnt_joiner.cpp
  src/streaming_asr.cpp
)

add_executable(zipformer_asr ${SOURCES})

target_include_directories(zipformer_asr PRIVATE
  ${SNDFILE_INCLUDE_DIRS}
)

target_link_libraries(zipformer_asr
  ${SNDFILE_LIBRARIES}
  sentencepiece
)

if(BUILD_RKNN)
  target_link_libraries(zipformer_asr rknnrt)
endif()

install(TARGETS zipformer_asr DESTINATION bin)
