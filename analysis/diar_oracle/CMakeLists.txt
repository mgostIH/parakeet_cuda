cmake_minimum_required(VERSION 3.26)
project(diar_oracle LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
get_filename_component(SRC ${CMAKE_CURRENT_SOURCE_DIR}/../../nemo-speech-src ABSOLUTE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE ON CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(${SRC}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)
set(GGML_DEPENDENCIES ggml ggml-base ggml-cpu)
set(NEMO_SPEECH_GGML_PATCHED OFF)
add_subdirectory(${SRC}/src/runtime/ggml ${CMAKE_BINARY_DIR}/runtime)
add_executable(diar_oracle main.cpp
 ${SRC}/src/asr/diar/sortformer_model.cpp
 ${SRC}/src/asr/diar/aosc_state.cpp
 ${SRC}/src/asr/diar/diar_pipeline.cpp
 ${SRC}/src/asr/features/fe.cpp
 ${SRC}/src/asr/encoder/rope_transformer.cpp
 ${SRC}/src/asr/encoder/transformer_encoder.cpp
 ${SRC}/src/asr/encoder/fastconformer.cpp
 ${SRC}/src/asr/encoder/rel_pos_attention.cpp)
target_include_directories(diar_oracle PRIVATE ${SRC}/src/asr ${SRC}/src/asr/features ${SRC}/src/asr/diar ${SRC}/src/asr/encoder ${SRC}/src/common)
target_link_libraries(diar_oracle PRIVATE nemo_speech_runtime_ggml)
target_include_directories(nemo_speech_runtime_ggml PRIVATE ${SRC}/src/common ${SRC}/ggml/src)
add_library(diar_state_reference SHARED state.cpp ${SRC}/src/asr/diar/aosc_state.cpp)
target_include_directories(diar_state_reference PRIVATE ${SRC}/src/asr/diar ${SRC}/src/asr/encoder ${SRC}/src/runtime/ggml ${SRC}/src/asr ${SRC}/ggml/include)
target_include_directories(diar_state_reference PRIVATE ${SRC}/src/common)
