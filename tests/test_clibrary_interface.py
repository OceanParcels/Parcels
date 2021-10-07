import pytest
import gc
import os
from parcels.tools import logger  # noqa
import ctypes

from parcels.compilation.codeinterface import LibraryRegisterC, InterfaceC
from parcels.compilation.codecompiler import GNUCompiler_MS, GNUCompiler_SS
from parcels.tools import get_package_dir, get_cache_dir

BUILD_LIB = 0
COMPILE_LIB = 1
LOAD_LIB = 2
EXECUTE_LIB = 3
UNLOAD_LIB = 4
REMOVE_LIB = 5


@pytest.fixture(name="testf_cstring")
def testf_cstring_fixture():
    _string_ = "#include \"testf.h\"\n" \
               "void func(int n) {\n" \
               "  printf(\"Transfer number: %d\", n);\n" \
               "}\n" \
               "\n" \
               "int get_val(void) {\n" \
               "  return 5;\n" \
               "}\n"
    return _string_


@pytest.fixture(name="testf_hstring")
def testf_hstring_fixture():
    _string_ = "#ifndef _ZIGGURAT_H\n" \
               "#define _ZIGGURAT_H\n" \
               "#ifdef __cplusplus\n" \
               "extern \"C\" {\n" \
               "#endif\n" \
               "#include <stdio.h>\n" \
               "void func(int n);\n" \
               "int get_val(void);\n" \
               "#ifdef __cplusplus\n" \
               "}\n" \
               "#endif\n" \
               "#endif\n"
    return _string_


@pytest.mark.parametrize('compiler', [GNUCompiler_SS, GNUCompiler_MS])
@pytest.mark.parametrize("call_gc", [True, False])
def test_clibrary_creation_teardown(compiler, call_gc):
    my_obj = LibraryRegisterC()
    libs = "node"  # if compiler is GNUCompiler_SS else ["node",]
    ccompiler = compiler(cppargs=[], incdirs=[os.path.join(get_package_dir(), 'include'), os.path.join(get_package_dir(), 'nodes'), "."], libdirs=[".", get_cache_dir()])
    my_interface = InterfaceC(libs, ccompiler, get_package_dir())
    del ccompiler
    del libs
    del my_interface
    del my_obj
    if call_gc:
        gc.collect()


@pytest.mark.parametrize('compiler', [GNUCompiler_SS, GNUCompiler_MS])
@pytest.mark.parametrize("manual_interface_teardown", [True, False])
@pytest.mark.parametrize("call_gc", [True, False])
def test_clibrary_add_remove_entry(compiler, manual_interface_teardown, call_gc):
    my_obj = LibraryRegisterC()
    libs = "node"  # if compiler is GNUCompiler_SS else ["node",]
    ccompiler = compiler(cppargs=[], incdirs=[os.path.join(get_package_dir(), 'include'), os.path.join(get_package_dir(), 'nodes'), "."], libdirs=[".", get_cache_dir()])
    my_interface = InterfaceC(libs, ccompiler, get_package_dir())
    my_obj.add_entry("node", my_interface)
    assert my_obj.get("node") == my_interface
    if manual_interface_teardown:
        my_obj.remove("node")
    del libs
    del ccompiler
    del my_interface
    del my_obj
    if call_gc:
        gc.collect()


@pytest.mark.parametrize('compiler', [GNUCompiler_SS, GNUCompiler_MS])
@pytest.mark.parametrize('stages_done', [BUILD_LIB, COMPILE_LIB, LOAD_LIB, EXECUTE_LIB, UNLOAD_LIB, REMOVE_LIB])
@pytest.mark.parametrize("manual_interface_teardown", [True, False])
@pytest.mark.parametrize("call_gc", [True, False])
def test_clibrary_test_custom_clib(testf_cstring, testf_hstring, compiler, stages_done, manual_interface_teardown, call_gc):
    with open(os.path.join(get_cache_dir(), "testf.c"), "w") as f:
        f.write(testf_cstring)
    with open(os.path.join(get_cache_dir(), "testf.h"), "w") as f:
        f.write(testf_hstring)
    my_obj = LibraryRegisterC()
    libnames = "testf"  # if compiler is GNUCompiler_SS else ["testf",]
    ccompiler = compiler(cppargs=[], incdirs=[os.path.join(get_cache_dir()), "."], libdirs=[".", get_cache_dir()])
    my_interface = InterfaceC(libnames, ccompiler, get_cache_dir())
    my_obj.add_entry("testf", my_interface)
    assert my_obj.is_created("testf")
    assert my_obj.get("testf") == my_interface
    if stages_done > BUILD_LIB:
        my_interface.compile_library()
        assert my_obj.is_compiled("testf")
    if stages_done > COMPILE_LIB:
        my_obj.load("testf", get_cache_dir())
        assert my_obj.is_loaded("testf")
    if stages_done > LOAD_LIB:
        testf_lib = my_obj.get("testf")
        function_param_array = [{"name": "func", "return": None, "arguments": [ctypes.c_int, ]},
                                {"name": "get_val", "return": ctypes.c_int, "arguments": None}]
        funcs = testf_lib.load_functions(function_param_array)
        func_testf = funcs["func"]
        func_getval = funcs["get_val"]
        func_testf(2)
        result = func_getval()
        assert result == 5
    if stages_done > EXECUTE_LIB:
        my_obj.unload("testf")
        assert not my_obj.is_loaded("testf")
    if stages_done > UNLOAD_LIB:
        my_obj.remove("testf")
        assert not my_obj.is_created("testf")
    if manual_interface_teardown:
        my_obj.clear()
    del my_interface
    del my_obj
    if call_gc:
        gc.collect()

# TODO
# - compile the self-written and the node-interface collectively
# - compile c/h-string-written code and call the functions
# - create a class that calls the c/h-functions, and test the register-/deregister functions
