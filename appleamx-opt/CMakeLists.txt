get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRArithDialect
        MLIROptLib
        MLIRAppleAMX
        )
add_llvm_executable(appleamx-opt appleamx-opt.cpp)

llvm_update_compile_flags(appleamx-opt)
target_link_libraries(appleamx-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(appleamx-opt)
