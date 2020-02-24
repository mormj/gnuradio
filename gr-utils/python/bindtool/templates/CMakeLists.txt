install(FILES
    generic_python_hpp.mako
    generic_python_h.mako
    generic_python_cc.mako
    license.mako
    python_bindings_cpp.mako
    # types.py
    DESTINATION ${GR_PYTHON_DIR}/gnuradio/bindtool/templates
)
