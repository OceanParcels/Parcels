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


class CodeGenerator(ast.NodeTransformer):

    def __init__(self, grid, Particle):
        self.grid = grid
        self.Particle = Particle

    def generate(self, pyfunc):
        # Traverse the tree, starting from the root of the function
        self.py_ast = ast.parse(inspect.getsource(pyfunc.func_code))
        self.py_ast = self.py_ast.body[0]

        ccode = self.visit(self.py_ast.body[1])
        return ccode

    def visit_Name(self, node):
        """Catches any mention of intrinsic variable names, such as
        'particle' or 'grid' and inserts our placeholder objects"""
        if node.id == 'grid':
            return GridNode(self.grid, ccode='grid')
        elif node.id == 'particle':
            return ParticleNode(self.Particle, ccode='particle')
        else:
            node.ccode = node.id
        return node

    def visit_Attribute(self, node):
        node.value = ast.NodeTransformer.visit(self, node.value)
        if isinstance(node.value, IntrinsicNode):
            return getattr(node.value, node.attr)
        else:
            return node

    def visit_Assign(self, node):
        t = self.visit(node.targets[0])
        v = self.visit(node.value)
        return c.Statement("%s = %s" % (t.ccode, v.ccode))

    def visit_Index(self, node):
        return self.visit(node.value)

    def visit_Tuple(self, node):
        return tuple([self.visit(e) for e in node.elts])

    def visit_Subscript(self, node):
        v = self.visit(node.value)
        s = self.visit(node.slice)
        if isinstance(v, FieldNode):
            return v.__getitem__(s)
        elif isinstance(v, IntrinsicNode):
            raise NotImplementedError("Subscript not implemented for object type %s"
                                      % type(v).__name__)
