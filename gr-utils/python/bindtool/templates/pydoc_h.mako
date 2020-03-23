##
## Copyright 2020 Free Software Foundation, Inc.
##
## This file is part of GNU Radio
##
## SPDX-License-Identifier: GPL-3.0-or-later
##
##
<%
    namespace = header_info['namespace']
    modname = header_info['module_name']
%>\
\
#include "pydoc_macros.h"
\
${render_namespace(namespace=namespace, modname=[modname])}

<%def name='render_docstring_const(modname,names,docstring="")'>
static const char *__doc_${'_'.join(modname+names)} =
R"doc(${docstring})doc";
</%def> \
<%def name='render_namespace(namespace, modname)'>
<%
    classes=namespace['classes'] if 'classes' in namespace else []
    free_functions=namespace['free_functions'] if 'free_functions' in namespace else []
    free_enums = namespace['enums'] if 'enums' in namespace else []
    subnamespaces = namespace['namespaces'] if 'namespaces' in namespace else []
%>\
\
% for cls in classes:
<%
        member_functions = cls['member_functions'] if 'member_functions' in cls else []
        constructors = cls['constructors'] if 'constructors' in cls else []
        class_enums = cls['enums'] if 'enums' in cls else []
        class_variables = cls['variables'] if 'variables' in cls else []
%>
\
${render_docstring_const(modname,[cls['name']],cls['docstring'] if 'docstring' in cls else "")}
\
% for cotr in constructors:
${render_docstring_const(modname,[cls['name'],cotr['name'],str(loop.index)],cotr['docstring'] if 'docstring' in cotr else "")}
% endfor ## constructors
\
% for fcn in member_functions:
${render_docstring_const(modname,[cls['name'],fcn['name']],fcn['docstring'] if 'docstring' in fcn else "")}
% endfor ## member_functions
% endfor ## classes
\
% if free_functions:
% for fcn in free_functions:
${render_docstring_const(modname,[fcn['name']],fcn['docstring'] if 'docstring' in fcn else "")}
% endfor ## free_functions
% endif ## free_functions
\
% for sns in subnamespaces:
<%  
  submod_name = sns['name'].split('::')[-1]
%>
${render_namespace(namespace=sns,modname=modname+[submod_name])}
% endfor
</%def>
