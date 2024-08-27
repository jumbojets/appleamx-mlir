llvm_canonicalize_cmake_booleans(
        MLIR_ENABLE_BINDINGS_PYTHON
)

configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(APPLEAMX_TEST_DEPENDS
        FileCheck count not
        appleamx-opt
        )

add_lit_testsuite(check-appleamx "Running the appleamx regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${APPLEAMX_TEST_DEPENDS}
        )
set_target_properties(check-appleamx PROPERTIES FOLDER "Tests")

add_lit_testsuites(APPLEAMX ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${APPLEAMX_TEST_DEPENDS})
