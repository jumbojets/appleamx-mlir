add_mlir_dialect(Ops appleamx)
add_mlir_doc(Dialect Dialect AppleAMX/ -gen-dialect-doc)
add_mlir_doc(Ops Ops AppleAMX/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRPassesIncGen)
