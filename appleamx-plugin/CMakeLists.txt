add_llvm_library(AppleAMXPlugin
        # BUILDTREE_ONLY is only for testing purposes
        MODULE BUILDTREE_ONLY
        appleamx-plugin.cpp

        DEPENDS
        MLIRAppleAMX
        PLUGIN_TOOL
        mlir-opt

        LINK_LIBS
        MLIRAppleAMX
        )
target_include_directories(
        AppleAMXPlugin
        PRIVATE
        "${APPLEAMX_BINARY_DIR}/include"
)
