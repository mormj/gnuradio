install(FILES
    generic_python_hpp.mako
    generic_python_h.mako
    generic_python_cc.mako
    CMakeLists.txt.mako
    license.mako
    python_bindings_cc.mako
    # types.py
    DESTINATION ${GR_PYTHON_DIR}/gnuradio/bindtool/templates
)
