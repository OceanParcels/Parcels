import ast
import inspect


class IntrinsicNode(ast.AST):

    def __init__(self, obj, ccode):
        self.obj = obj
        self.ccode = ccode

    def __getattr__(self, attr):
        return IntrinsicNode(getattr(self.obj, attr),
                             ccode="%s.%s" % (self.ccode, attr))


class IntrinsicParticle(IntrinsicNode):
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

        ccode = self.visit(self.py_ast)
        return ast.dump(ccode)

    def visit_Name(self, node):
        """Catches any mention of intrinsic variable names, such as
        'particle' or 'grid' and inserts our placeholder objects"""
        if node.id == 'grid':
            return IntrinsicNode(self.grid, ccode='grid')
        elif node.id == 'particle':
            return IntrinsicParticle(self.Particle, ccode='particle')
        return node

    def visit_Attribute(self, node):
        node.value = ast.NodeTransformer.visit(self, node.value)
        if isinstance(node.value, IntrinsicNode):
            return getattr(node.value, node.attr)
        else:
            return node
