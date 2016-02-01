import ast
import cgen as c
from collections import OrderedDict


class IntrinsicNode(ast.AST):
    def __init__(self, obj, ccode):
        self.obj = obj
        self.ccode = ccode


class GridNode(IntrinsicNode):
    def __getattr__(self, attr):
        return FieldNode(getattr(self.obj, attr),
                         ccode="%s->%s" % (self.ccode, attr))


class FieldNode(IntrinsicNode):
    def __getitem__(self, attr):
        return IntrinsicNode(None, ccode=self.obj.ccode_subscript(*attr))


class ParticleAttributeNode(IntrinsicNode):
    def __init__(self, obj, attr):
        self.obj = obj
        self.attr = attr
        self.ccode = "%s->%s" % (obj.ccode, attr)
        self.ccode_index_var = None

        if self.attr == 'lon':
            self.ccode_index_var = "%s->%s" % (self.obj.ccode, "xi")
        elif self.attr == 'lat':
            self.ccode_index_var = "%s->%s" % (self.obj.ccode, "yi")

    @property
    def pyast_index_update(self):
        pyast = ast.Assign()
        pyast.targets = [IntrinsicNode(None, ccode=self.ccode_index_var)]
        pyast.value = IntrinsicNode(None, ccode=self.ccode_index_update)
        return pyast

    @property
    def ccode_index_update(self):
        """C-code for the index update requires after updating p.lon/p.lat"""
        if self.attr == 'lon':
            return "search_linear_float(%s, %s, U->xdim, U->lat)" \
                % (self.ccode, self.ccode_index_var)
        if self.attr == 'lat':
            return "search_linear_float(%s, %s, U->ydim, U->lon)" \
                % (self.ccode, self.ccode_index_var)
        return ""


class ParticleNode(IntrinsicNode):
    def __getattr__(self, attr):
        if attr in self.obj.var_types:
            return ParticleAttributeNode(self, attr)
        else:
            raise AttributeError("""Particle type %s does not define attribute "%s".
Please add '%s' to %s.users_vars or define an appropriate sub-class."""
                                 % (self.obj, attr, attr, self.obj))


class IntrinsicTransformer(ast.NodeTransformer):
    """AST transformer that catches any mention of intrinsic variable
    names, such as 'particle' or 'grid', inserts placeholder objects
    and propagates attribute access."""

    def __init__(self, grid, ptype):
        self.grid = grid
        self.ptype = ptype

    def visit_Name(self, node):
        if node.id == 'grid':
            return GridNode(self.grid, ccode='grid')
        elif node.id == 'particle':
            return ParticleNode(self.ptype, ccode='particle')
        else:
            return node

    def visit_Attribute(self, node):
        node.value = self.visit(node.value)
        if isinstance(node.value, IntrinsicNode):
            return getattr(node.value, node.attr)
        else:
            raise NotImplementedError("Cannot propagate attribute access to C-code")

    def visit_AugAssign(self, node):
        node.target = self.visit(node.target)
        node.op = self.visit(node.op)
        node.value = self.visit(node.value)

        # Capture p.lat/p.lon updates and insert p.xi/p.yi updates
        if isinstance(node.target, ParticleAttributeNode) \
           and node.target.ccode_index_var is not None:
            node = [node, node.target.pyast_index_update]
        return node

    def visit_Assign(self, node):
        node.targets = [self.visit(t) for t in node.targets]
        node.value = self.visit(node.value)

        # Capture p.lat/p.lon updates and insert p.xi/p.yi updates
        if isinstance(node.targets[0], ParticleAttributeNode) \
           and node.targets[0].ccode_index_var is not None:
            node = [node, node.targets[0].pyast_index_update]
        return node


class TupleSplitter(ast.NodeTransformer):
    """AST transformer that detects and splits Pythonic tuple
    assignments into multiple statements for conversion to C."""

    def visit_Assign(self, node):
        if isinstance(node.targets[0], ast.Tuple) \
           and isinstance(node.value, ast.Tuple):
            t_elts = node.targets[0].elts
            v_elts = node.value.elts
            if len(t_elts) != len(v_elts):
                raise AttributeError("Tuple lenghts in assignment do not agree")
            node = [ast.Assign() for _ in t_elts]
            for n, t, v in zip(node, t_elts, v_elts):
                n.targets = [t]
                n.value = v
        return node


class KernelGenerator(ast.NodeVisitor):
    """Code generator class that translates simple Python kernel
    functions into C functions by populating and accessing the `ccode`
    attriibute on nodes in the Python AST."""

    # Intrinsic variables that appear as function arguments
    kernel_vars = ['particle', 'grid', 'time', 'dt']

    def __init__(self, grid, ptype):
        self.grid = grid
        self.ptype = ptype
        self.field_args = OrderedDict()

    def generate(self, py_ast, funcvars):
        # Untangle Pythonic tuple-assignment statements
        py_ast = TupleSplitter().visit(py_ast)

        # Replace occurences of intrinsic objects in Python AST
        transformer = IntrinsicTransformer(self.grid, self.ptype)
        py_ast = transformer.visit(py_ast.body[0])

        # Generate C-code for all nodes in the Python AST
        self.visit(py_ast)
        self.ccode = py_ast.ccode

        # Insert variable declarations for non-instrinsics
        for kvar in self.kernel_vars:
            if kvar in funcvars:
                funcvars.remove(kvar)
        self.ccode.body.insert(0, c.Value("float", ", ".join(funcvars)))

        return self.ccode

    def visit_FunctionDef(self, node):
        # Generate "ccode" attribute by traversing the Python AST
        for stmt in node.body:
            self.visit(stmt)

        # Create function declaration and argument list
        decl = c.Static(c.DeclSpecifier(c.Value("void", node.name), spec='inline'))
        args = [c.Pointer(c.Value(self.ptype.name, "particle")),
                c.Value("double", "time"), c.Value("float", "dt")]
        for field, _ in self.field_args.items():
            args += [c.Pointer(c.Value("CField", "%s" % field))]

        # Create function body as C-code object
        body = c.Block([stmt.ccode for stmt in node.body])
        node.ccode = c.FunctionBody(c.FunctionDeclaration(decl, args), body)

    def visit_Name(self, node):
        """Catches any mention of intrinsic variable names, such as
        'particle' or 'grid' and inserts our placeholder objects"""
        node.ccode = node.id

    def visit_Assign(self, node):
        self.visit(node.targets[0])
        self.visit(node.value)
        node.ccode = c.Assign(node.targets[0].ccode, node.value.ccode)

    def visit_AugAssign(self, node):
        self.visit(node.target)
        self.visit(node.op)
        self.visit(node.value)
        node.ccode = c.Statement("%s %s= %s" % (node.target.ccode,
                                                node.op.ccode,
                                                node.value.ccode))

    def visit_Index(self, node):
        self.visit(node.value)
        node.ccode = node.value.ccode

    def visit_Tuple(self, node):
        for e in node.elts:
            self.visit(e)
        node.ccode = tuple([e.ccode for e in node.elts])

    def visit_Subscript(self, node):
        self.visit(node.value)
        self.visit(node.slice)
        if isinstance(node.value, FieldNode):
            node.ccode = node.value.__getitem__(node.slice.ccode).ccode
        elif isinstance(node.value, IntrinsicNode):
            raise NotImplementedError("Subscript not implemented for object type %s"
                                      % type(node.value).__name__)

    def visit_BinOp(self, node):
        self.visit(node.left)
        self.visit(node.op)
        self.visit(node.right)
        node.ccode = "(%s %s %s)" % (node.left.ccode, node.op.ccode, node.right.ccode)

    def visit_Add(self, node):
        node.ccode = "+"

    def visit_Sub(self, node):
        node.ccode = "-"

    def visit_Mult(self, node):
        node.ccode = "*"

    def visit_Div(self, node):
        node.ccode = "/"

    def visit_Num(self, node):
        node.ccode = str(node.n)

    def visit_FieldNode(self, node):
        """Record intrinsic fields used in kernel"""
        self.field_args[node.obj.name] = node.obj


class LoopGenerator(object):
    """Code generator class that adds type definitions and the outer
    loop around kernel functions to generate compilable C code."""

    def __init__(self, grid, ptype=None):
        self.grid = grid
        self.ptype = ptype

    def generate(self, funcname, field_args, kernel_ast):
        ccode = []

        # Add include for Parcels header
        ccode += [str(c.Include("parcels.h", system=False))]

        # Generate type definition for particle type
        vdecl = [c.POD(dtype, var) for var, dtype in self.ptype.var_types.items()]
        ccode += [str(c.Typedef(c.GenerableStruct("", vdecl, declname=self.ptype.name)))]

        # Insert kernel code
        ccode += [str(kernel_ast)]

        # Generate outer loop for repeated kernel invocation
        args = [c.Value("int", "num_particles"),
                c.Pointer(c.Value(self.ptype.name, "particles")),
                c.Value("int", "timesteps"), c.Value("double", "time"),
                c.Value("float", "dt")]
        for field, _ in field_args.items():
            args += [c.Pointer(c.Value("CField", "%s" % field))]
        fargs_str = ", ".join(field_args.keys())
        loop_body = [c.Statement("%s(&(particles[p]), time, dt, %s)" %
                                 (funcname, fargs_str))]
        ploop = c.For("p = 0", "p < num_particles", "++p", c.Block(loop_body))
        tloop = c.For("t = 0", "t < timesteps", "++t",
                      c.Block([ploop, c.Statement("time += (double)dt")]))
        fbody = c.Block([c.Value("int", "p, t"), tloop])
        fdecl = c.FunctionDeclaration(c.Value("void", "particle_loop"), args)
        ccode += [str(c.FunctionBody(fdecl, fbody))]
        return "\n\n".join(ccode)
