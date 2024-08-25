add_mlir_dialect_library(MLIRAppleAMX
        Types.cpp
        Dialect.cpp
        Ops.cpp
        Passes.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/AppleAMX

        DEPENDS
        MLIROpsIncGen
        MLIRPassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        )
