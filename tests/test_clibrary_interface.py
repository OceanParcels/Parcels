import gc
import os
import pytest
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
    _string_ = "#ifndef _TESTFUNC_H\n" \
               "#define _TESTFUNC_H\n" \
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
    ccompiler = compiler(cppargs=[], incdirs=[os.path.join(get_package_dir(), 'include'), os.path.join(get_package_dir(), 'nodes'), "."], libdirs=[".", get_cache_dir()])
    my_interface = InterfaceC("node", ccompiler, os.path.join(get_package_dir(), "nodes"))
    del ccompiler
    del my_interface
    del my_obj
    if call_gc:
        gc.collect()


@pytest.mark.parametrize('compiler', [GNUCompiler_SS, GNUCompiler_MS])
@pytest.mark.parametrize("manual_interface_teardown", [True, False])
@pytest.mark.parametrize("call_gc", [True, False])
def test_clibrary_add_remove_entry(compiler, manual_interface_teardown, call_gc):
    my_obj = LibraryRegisterC()
    ccompiler = compiler(cppargs=[], incdirs=[os.path.join(get_package_dir(), 'include'), os.path.join(get_package_dir(), 'nodes'), "."], libdirs=[".", get_cache_dir()])
    my_interface = InterfaceC("node", ccompiler, os.path.join(get_package_dir(), "nodes"))
    my_obj.add_entry("node", my_interface)
    assert my_obj.get("node") == my_interface
    if manual_interface_teardown:
        my_obj.remove("node")
    del ccompiler
    del my_interface
    del my_obj
    if call_gc:
        gc.collect()


@pytest.mark.parametrize('compiler', [GNUCompiler_SS, GNUCompiler_MS])
@pytest.mark.parametrize('stages_done', [BUILD_LIB, COMPILE_LIB, LOAD_LIB, EXECUTE_LIB, UNLOAD_LIB, REMOVE_LIB])
@pytest.mark.parametrize("manual_interface_teardown", [True, False])
@pytest.mark.parametrize("call_gc", [True, False])
def test_clibrary_custom_clib(testf_cstring, testf_hstring, compiler, stages_done, manual_interface_teardown, call_gc):
    testf_c_file = os.path.join(get_cache_dir(), "testf.c")
    testf_h_file = os.path.join(get_cache_dir(), "testf.h")
    with open(testf_c_file, "w") as f:
        f.write(testf_cstring)
    with open(testf_h_file, "w") as f:
        f.write(testf_hstring)
    my_obj = LibraryRegisterC()
    ccompiler = compiler(cppargs=[], incdirs=[os.path.join(get_cache_dir()), "."], libdirs=[".", get_cache_dir()])
    my_interface = InterfaceC("testf", ccompiler, get_cache_dir())
    my_obj.add_entry("testf", my_interface)
    assert my_obj.iscreated("testf")
    assert my_obj.get("testf") == my_interface
    if stages_done > BUILD_LIB:
        my_interface.compile_library()
        assert my_obj.iscompiled("testf")
    if stages_done > COMPILE_LIB:
        my_obj.load("testf")
        assert my_obj.isloaded("testf")
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
        assert not my_obj.isloaded("testf")
    if stages_done > UNLOAD_LIB:
        my_obj.remove("testf")
        assert not my_obj.iscreated("testf")
    if manual_interface_teardown:
        my_obj.clear()
    del my_interface
    del my_obj
    if os.path.exists(testf_c_file):
        os.remove(testf_c_file)
    if os.path.exists(testf_h_file):
        os.remove(testf_h_file)
    if call_gc:
        gc.collect()


def test_clibrary_multilib_collective_compile(testf_cstring, testf_hstring):
    testf_c_file = os.path.join(get_cache_dir(), "testf.c")
    testf_h_file = os.path.join(get_cache_dir(), "testf.h")
    with open(testf_c_file, "w") as f:
        f.write(testf_cstring)
    with open(testf_h_file, "w") as f:
        f.write(testf_hstring)
    my_obj = LibraryRegisterC()
    ccompiler_testf = GNUCompiler_MS(cppargs=[],
                                     incdirs=[os.path.join(get_package_dir(), 'include'), os.path.join(get_package_dir(), 'nodes'), os.path.join(get_cache_dir()), "."],
                                     libdirs=[".", get_cache_dir()],
                                     libs=[])
    interface_testf = InterfaceC(["node", "testf"], ccompiler_testf, [os.path.join(get_package_dir(), "nodes"), get_cache_dir()])
    my_obj.add_entry("testf", interface_testf)
    assert my_obj.iscreated("testf")
    assert my_obj.get("testf") == interface_testf
    interface_testf.compile_library()
    assert my_obj.iscompiled("testf")
    my_obj.load("testf")
    assert my_obj.isloaded("testf")
    testf_lib = my_obj.get("testf")
    function_param_array = [{"name": "func", "return": None, "arguments": [ctypes.c_int, ]},
                            {"name": "get_val", "return": ctypes.c_int, "arguments": None}]
    funcs = testf_lib.load_functions(function_param_array)
    func_testf = funcs["func"]
    func_getval = funcs["get_val"]
    func_testf(2)
    result = func_getval()
    assert result == 5
    my_obj.unload("testf")
    assert not my_obj.isloaded("testf")
    my_obj.remove("testf")
    assert not my_obj.iscreated("testf")
    my_obj.clear()
    del interface_testf
    del my_obj
    if os.path.exists(testf_c_file):
        os.remove(testf_c_file)
    if os.path.exists(testf_h_file):
        os.remove(testf_h_file)


def test_clibrary_multilib_separate_compile(testf_cstring, testf_hstring):
    testf_c_file = os.path.join(get_cache_dir(), "testf.c")
    testf_h_file = os.path.join(get_cache_dir(), "testf.h")
    with open(testf_c_file, "w") as f:
        f.write(testf_cstring)
    with open(testf_h_file, "w") as f:
        f.write(testf_hstring)
    my_obj = LibraryRegisterC()
    ccompiler_node = GNUCompiler_SS(cppargs=[], incdirs=[os.path.join(get_package_dir(), 'include'), os.path.join(get_package_dir(), 'nodes'), "."], libdirs=[".", get_cache_dir()])
    interface_node = InterfaceC("node", ccompiler_node, src_dir=os.path.join(get_package_dir(), "nodes"))
    ccompiler_testf = GNUCompiler_SS(cppargs=[], incdirs=[os.path.join(get_cache_dir()), "."], libdirs=[".", get_cache_dir()])
    interface_testf = InterfaceC("testf", ccompiler_testf, src_dir=get_cache_dir())
    my_obj.add_entry("node", interface_node)
    my_obj.add_entry("testf", interface_testf)
    assert my_obj.iscreated("node")
    assert my_obj.iscreated("testf")
    assert my_obj.get("node") == interface_node
    assert my_obj.get("testf") == interface_testf
    interface_node.compile_library()
    interface_testf.compile_library()
    assert my_obj.iscompiled("node")
    assert my_obj.iscompiled("testf")
    my_obj.load("node")
    my_obj.load("testf")
    assert my_obj.isloaded("node")
    assert my_obj.isloaded("testf")
    testf_lib = my_obj.get("testf")
    function_param_array = [{"name": "func", "return": None, "arguments": [ctypes.c_int, ]},
                            {"name": "get_val", "return": ctypes.c_int, "arguments": None}]
    funcs = testf_lib.load_functions(function_param_array)
    func_testf = funcs["func"]
    func_getval = funcs["get_val"]
    func_testf(2)
    result = func_getval()
    assert result == 5
    my_obj.unload("node")
    my_obj.unload("testf")
    assert not my_obj.isloaded("node")
    assert not my_obj.isloaded("testf")
    my_obj.remove("node")
    my_obj.remove("testf")
    assert not my_obj.iscreated("node")
    assert not my_obj.iscreated("testf")
    my_obj.clear()
    del interface_testf
    del my_obj
    if os.path.exists(testf_c_file):
        os.remove(testf_c_file)
    if os.path.exists(testf_h_file):
        os.remove(testf_h_file)


def test_clibrary_inner_class_registration(testf_cstring, testf_hstring):

    my_obj = LibraryRegisterC()
    testf_c_file = os.path.join(get_cache_dir(), "testf.c")
    testf_h_file = os.path.join(get_cache_dir(), "testf.h")
    with open(testf_c_file, "w") as f:
        f.write(testf_cstring)
    with open(testf_h_file, "w") as f:
        f.write(testf_hstring)

    class TestClass(object):
        registered = False

        def __init__(self, c_lib_register):
            libname = "testf"
            src_dir = get_cache_dir()
            if not c_lib_register.iscreated(libname) or not c_lib_register.iscompiled(libname):
                cppargs = []
                ccompiler = GNUCompiler_SS(cppargs=cppargs, incdirs=[os.path.join(get_package_dir(), 'include'), os.path.join(get_package_dir(), 'nodes'), "."], libdirs=[".", get_cache_dir()])
                my_interface = InterfaceC(libname, ccompiler, src_dir)
                c_lib_register.add_entry(libname, my_interface)
            if not c_lib_register.isloaded(libname):
                c_lib_register.load(libname)
            c_lib_register.register(libname, close_callback=self.close_c_cb)
            self.c_lib_register_ref = c_lib_register
            self.registered = True
            self.c_interface = self.c_lib_register_ref.get(libname)
            function_param_array = [{"name": "func", "return": None, "arguments": [ctypes.c_int, ]},
                                    {"name": "get_val", "return": ctypes.c_int, "arguments": None}]
            c_funcs = self.c_interface.load_functions(function_param_array)
            if c_funcs is None:
                c_lib_register.deregister(libname=libname)
                c_lib_register.unload(libname=libname)
                c_lib_register.load(libname)
                c_lib_register.register(libname, close_callback=self.close_c_cb)
            assert c_funcs is not None, "Loading 'node' library failed."
            self.func_testf = c_funcs["func"]
            self.func_getval = c_funcs["get_val"]

        def close(self):
            self.c_lib_register_ref.deregister(libname="testf")
            if self.c_interface.register_count <= 0:
                self.c_interface.close()

        def execute(self):
            if self.registered and self.c_lib_register_ref is not None and self.func_testf is not None:
                self.func_testf(2)
                result = self.func_getval()
                assert result == 5

        def close_c_cb(self):
            self.func_testf = None
            self.func_getval = None
            self.c_interface = None
            self.c_lib_register_ref = None
            self.registered = False

    obj1 = TestClass(my_obj)
    obj2 = TestClass(my_obj)
    obj1.execute()
    obj2.execute()
    obj1.close()
    obj1.close()
    if my_obj.isloaded("testf"):
        my_obj.unload("testf")
    assert not my_obj.isloaded("testf")
    if my_obj.iscreated("testf"):
        my_obj.remove("testf")
    assert not my_obj.iscreated("testf")
    my_obj.clear()
    del my_obj
    if os.path.exists(testf_c_file):
        os.remove(testf_c_file)
    if os.path.exists(testf_h_file):
        os.remove(testf_h_file)
