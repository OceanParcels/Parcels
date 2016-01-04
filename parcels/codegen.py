import ast
import inspect
import cgen as c


class IntrinsicNode(ast.AST):
    def __init__(self, obj, ccode):
        self.obj = obj
        self.ccode = ccode


class GridNode(IntrinsicNode):
    def __getattr__(self, attr):
        return FieldNode(getattr(self.obj, attr),
                         ccode="%s.%s" % (self.ccode, attr))


class FieldNode(IntrinsicNode):
    def __getitem__(self, attr):
        return IntrinsicNode(None, ccode=self.obj.ccode_subscript(*attr))


class ParticleNode(IntrinsicNode):
    def __getattr__(self, attr):
        if attr in self.obj.base_vars or attr in self.obj.user_vars:
            return IntrinsicNode(None, ccode="%s.%s" % (self.ccode, attr))
        else:
            raise AttributeError("""Particle type %s does not define
attribute "%s".  Please add '%s' to %s.users_vars or define an appropriate sub-class."""
                                 % (self.obj, attr, attr, self.obj))


class IntrinsicTransformer(ast.NodeTransformer):
    """AST transformer that catches any mention of intrinsic variable
    names, such as 'particle' or 'grid', inserts placeholder objects
    and propagates attribute access"""

    def __init__(self, grid, Particle):
        self.grid = grid
        self.Particle = Particle

    def visit_Name(self, node):
        if node.id == 'grid':
            return GridNode(self.grid, ccode='grid')
        elif node.id == 'particle':
            return ParticleNode(self.Particle, ccode='particle')
        else:
            return node

    def visit_Attribute(self, node):
        node.value = ast.NodeTransformer.visit(self, node.value)
        if isinstance(node.value, IntrinsicNode):
            return getattr(node.value, node.attr)
        else:
            raise NotImplementedError("Cannot propagate attribute access to C-code")


class CodeGenerator(ast.NodeVisitor):

    def __init__(self, grid, Particle):
        self.grid = grid
        self.Particle = Particle

    def generate(self, pyfunc):
        # Parse the Python code into an AST
        self.py_ast = ast.parse(inspect.getsource(pyfunc.func_code))

        # Replace occurences of intrinsic objects in Python AST
        transformer = IntrinsicTransformer(self.grid, self.Particle)
        self.py_ast = transformer.visit(self.py_ast.body[0])

        # Generate C-code for all nodes in the Python AST
        self.visit(self.py_ast)

        return self.py_ast.ccode

    def visit_FunctionDef(self, node):
        for stmt in node.body:
            self.visit(stmt)
        node.ccode = c.Block([stmt.ccode for stmt in node.body])

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
        node.ccode = "%s %s %s" % (node.left.ccode, node.op.ccode, node.right.ccode)

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
