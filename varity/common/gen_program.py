
import gen_inputs
import id_generator
import gen_inputs
import cfg

from enum import Enum
import random
import subprocess

from random_functions import lucky, randomListChunk, veryLucky, generateMathExpression
from type_checking import getTypeString, isTypeReal, isTypeRealPointer, isTypeInt

# ================= Global State ===================
parallel_region_generated = False
inCriticalSection = None 
sectionId = 0
calledNodes = [] # stack of nodes created (ordered)
# ==================================================

# A wrapper for the data-sharing attributes for a parallel for-block. This
# object should only be necessary for ForLoopBlocks and OperationsBlocks, as
# the ForLoopBlock determines the attributes, while the OperationsBlocks needs
# the attributes to determine which sections are critical.
class DataSharingAttributes:
    def __init__(self, sharedVars=[], privateVars=[], firstPrivateVars=[]):
        self.sharedVars = sharedVars
        self.privateVars = privateVars 
        self.firstPrivateVars = firstPrivateVars 
        
    def getSharedVars(self):
        return list(self.sharedVars)

    def getPrivateVars(self):
        return list(self.privateVars)

    def getFirstPrivateVars(self):
        return (self.firstPrivateVars)

# Basic node in a tree
class Node:
    def __init__(self, code, left=None, right=None, isParallel=False,
                 dataSharingAttribs=DataSharingAttributes()):
        self.code = code
        self.left  = left
        self.right = right
        self.isParallel = isParallel
        self.dataSharingAttribs = dataSharingAttribs 

    def __str__(self):
        return str(self.code)

    def printCode(self) -> str:
        return "{};\n"
    
    # In case we want to get other node's attributes. Not sure if
    # will be relevant in final version.
    def getSharedVars(self):
        return self.dataSharingAttribs.getSharedVars()

    def getPrivateVars(self):
        return self.dataSharingAttribs.getPrivateVars()

    def getFirstPrivateVars(self):
        return self.dataSharingAttribs.getFirstPrivateVars()
    
    # If the node involves at least one parallel data-sharing attribute such as
    # "shared", then we will mark it as critical. **This is over-simplified
    # semantics, but it will work for now**
    def isCritical(self, varsToBeUsed):
        if self.isParallel:
            for v in varsToBeUsed:
                if v in self.getSharedVars():
                    return True

        return False

# Types of binary operations
class BinaryOperationType(Enum):
    add = 0
    sub = 1
    mul = 2
    div = 3
    
class BinaryOperation(Node):
    def __init__(self, code="", left=None, right=None):
        self.code = code
        self.left  = left
        self.right = right
    
    def generate(self):
        op = random.choice(list(BinaryOperationType))
        if op == BinaryOperationType.add:
            self.code = " + "
        elif op == BinaryOperationType.sub:
            self.code = " - "
        elif op == BinaryOperationType.mul:
            self.code = " * "
        elif op == BinaryOperationType.div:
            self.code = " / "

    def printCode(self) -> str:
        return self.code

class Expression(Node):
    rootNode = None
    def __init__(self, code="=", left=None, right=None, varToBeUsed=[], isParallel=True,
                 dataSharingAttribs=DataSharingAttributes()):
        import gen_math_exp
        self.left  = left
        self.right = right
        self.varToBeUsed = varToBeUsed
        self.usedVars = set(varToBeUsed)
        self.isParallel = isParallel
        
        if lucky():
            self.code = code
        else:
            self.code = "+"+code
            
        size = random.randrange(1, cfg.MAX_EXPRESSION_SIZE)

        lastOp = None
        mathExpTerminator = None
        while (size >= 1):
            if generateMathExpression():
                op = gen_math_exp.MathExpression()
                mathExpTerminator = True
                self.usedVars = self.usedVars.union(op.usedVars)
            else:
                op = BinaryOperation()
                op.generate()
                if lucky():
                    op.left = gen_inputs.InputGenerator.genInput()
                else:
                    op.left = id_generator.IdGenerator.get().generateRealID()
                    
                self.usedVars.add(op.left)

            
            if lastOp != None:
                lastOp.right = op
            else:
                self.rootNode = op
            lastOp = op
            size = size - 1


            # If we have a math expression, this is a terminator for the tree
            if mathExpTerminator == True:
                break
         
        if not mathExpTerminator:
            if lucky():
                lastOp.right= gen_inputs.InputGenerator.genInput()
            else:
                lastOp.right= id_generator.IdGenerator.get().generateRealID()
            
            self.usedVars.add(lastOp.right)
            

    def total(self, n):
        import gen_math_exp
        if n == None:
            print("NOT GOOD")
        elif isinstance(n, str):
            return n
        elif isinstance(n, gen_math_exp.MathExpression):
            return n.printCode()
        
        ret = Expression.total(self, n.left) + n.code + Expression.total(self, n.right)
        if lucky():
            return '(' + ret + ')'
        return ret

    def printCode(self, assignment=True) -> str:
        calledNodes.append("Expression")
        t = Expression.total(self, self.rootNode)
        if self.varToBeUsed != None:
            for v in self.varToBeUsed:
                op = BinaryOperation()
                op.generate()
                t = v + op.printCode() + t

        if assignment:
            p = ""
            #if self.isCritical():
            #    p = "#pragma omp critical\n"
            return p + "comp " + self.code + " " + t + ";"
        else:
            return t

class VariableDefinition(Node):
    def __init__(self, code=" = ", left=None, right=None, isPointer=False, isParallel=False,
                 dataSharingAttribs=DataSharingAttributes()):
        self.code = code
        self.right = right
        self.isPointer = isPointer
        self.isParallel = isParallel
        self.dataSharingAttribs = dataSharingAttribs
        self.usedVars = set()
        self.isCritical = False
        
        if isPointer == True:
            varName = id_generator.IdGenerator.get().generateRealID(True)
            self.usedVars.add(varName)
            self.left  = varName + "[i]"
        else:
            varName = id_generator.IdGenerator.get().generateTempRealID()
            self.usedVars.add(varName)
            print("Generate {}".format(varName))
            self.left = getTypeString() + " " + varName
            #if not self.isParallel:
            #    self.left  = getTypeString() + " " + varName 
            # Variable definitions in parallel blocks are "lifted" to the
            # top of the parent parallel block.
            #else:
            #    self.left  = varName 
        
        if lucky(): # constant definition
            self.right = gen_inputs.InputGenerator.genInput()
        else:
            self.right = Expression()
            
        if not isinstance(self.right, str):
            self.usedVars = self.usedVars.union(self.right.usedVars)
            
    def markCritical(self):
        self.isCritical = True

    def getVarName(self):
        if not self.isPointer:
#            if not self.isParallel:
#                return self.left.split(" ")[1]
#            return self.left
            return self.left.split(" ")[1]
        else:
            #print(self.left[:-3])
            return self.left
   
    def printCode(self) -> str:
        calledNodes.append("VariableDefinition")
        if isinstance(self.right, str):
            c = self.right
        else:
            c = self.right.printCode(False)
        
        p = ""
        # TODO(patrickjchap): Double-check this.
        #if self.isCritical():
        #    p = "#pragma omp critical\n"
            
        print("Left: " +  self.left)
        if not self.isPointer and self.isCritical:
            p += "// self.left: {}\n".format(self.left)
            return p + self.left.split(" ")[1] + self.code + c + ";\n"
        return p + self.left + self.code + c + ";\n"
    
    def size(self) -> int:
        if not self.isPointer:
            return 0
        

# A non-recursive block has only expressions (it does not have if-blocks or loop-blocks)
class OperationsBlock(Node):
    def __init__(self, level=1, code="", left=None, right=None, inLoop=False, recursive=True, isParallel=False,
                 dataSharingAttribs=DataSharingAttributes()):
        self.code = code
        self.left  = left
        self.right = right
        self.isParallel = isParallel
        self.level = level
        self.dataSharingAttribs = dataSharingAttribs
        self.usedVars = set()
        # These are variables defined in critical sections.
        self.definedCriticalVariables = set()
        
        # Defines the number of lines that the block will have
        lines = random.randrange(1, cfg.MAX_LINES_IN_BLOCK+1)
        assert lines > 0

        # In the block, we either have definitions of new variables or 
        # assigments to comp. The last line of the block will always be 
        # an assigment from an expression:
        #    comp = ...
        if lines == 1:
            self.left = [Expression(isParallel=isParallel)]
            self.usedVars.union(self.left[0].usedVars)
        else:
            i = 1
            varsToBeUsed = []
            l = []
            while(i <= lines):
                    
                if lucky() or i==lines: # expression with assigment
                    c = None
                    if len(varsToBeUsed) > 0:
                        c = Expression("=", None, None, varsToBeUsed[:],isParallel)
                        varsToBeUsed.clear()
                    else:
                        c = Expression(isParallel=isParallel)
                    l.append(c)
                    self.usedVars = self.usedVars.union(c.usedVars)
                    if i==lines:
                        break
                else:
                    if inLoop==True and lucky():
                        v = VariableDefinition(isPointer=True, isParallel=isParallel)
                        #v = VariableDefinition(isPointer=False, isParallel=isParallel)
                    else:
                        v = VariableDefinition(isParallel=isParallel)
                    l.append(v)
                    varsToBeUsed.append(v.getVarName())
                    self.usedVars.add(v.getVarName())

                i = i+1
            self.left = l

        # An operations block can also have if-conditions and loop blocks
        if recursive:
            nBlocks = random.randrange(0, cfg.MAX_SAME_LEVEL_BLOCKS+1)
            #nBlocks = 2
            for k in range(nBlocks):
                if lucky():
                    b = IfConditionBlock(recursive=False, isParallel=isParallel)
                else:
                    # If we are already in a parallel for-loop, we don't
                    # want to enter another.
                    if cfg.PARALLEL_PROG and not self.isParallel:
                        # Randomly start a parallel for-loop if we are not in one.
                        if lucky():
                            b = ForLoopBlock(recursive=False, isParallel=True, level=self.level+1)
                        # Mark the for-loop parallel if we are in a parallel block.
                        # We should be OK to not have nested parallel for-loops, as
                        # we have a global state check for this. Marking this parallel
                        # ensures we mark it critical if necessary!
                        else:
                            b = ForLoopBlock(recursive=False, isParallel=isParallel, level=self.level+1)
                    else:
                        b = ForLoopBlock(recursive=False, isParallel=isParallel, level=self.level+1)
                #print("1: ", self.usedVars)
                self.usedVars = self.usedVars.union(b.usedVars)
                self.left.append(b)
                #print("2: ", self.usedVars)
                
    def isLineCritical(self, line):
        if self.isParallel:
            for v in line.usedVars:
                if (v in self.dataSharingAttribs.getSharedVars() or v in self.definedCriticalVariables):
                    if isinstance(line, VariableDefinition):
                        self.definedCriticalVariables.add(line.getVarName())
                        line.markCritical()
                    return True
                
        return False
    
    def getCriticalBounds(self):
        low = None
        high = None 
        prevLineCritical = False
        for idx, l in enumerate(self. left):
            if self.isLineCritical(l):
                if not prevLineCritical and low == None:
                    #if isinstance(l, ForLoopBlock):
                    #    continue
                    low = idx 
                    high = idx 
                    prevLineCritical = True
                else:
                    high = idx
                    prevLineCritical = True
            else:
                prevLineCritical = False
                
        return low, high
            
                    
    def printCode(self) -> str:
        calledNodes.append("OperationsBlock")
        global inCriticalSection
        global sectionId
        sectionId = sectionId + 1
        self.id = sectionId
        ret = []
        prevLineCritical = False
        # TODO(patrickjchap): I think some of this low/high stuff is unnecessary now.
        # This causes printing to from O(n) to O(2n) from the OperationsBlock.
        low, high = self.getCriticalBounds()
        for idx, l in enumerate(self.left):
            p = ""
            # We assign the inCriticalSection to the specific operations block
            # that started the critical section so that we don't exit the critical
            # section prematurely in another block.
            if not inCriticalSection and low != None and low == idx:
                inCriticalSection = self.id 
                p = ""
                for var in self.definedCriticalVariables:
                    if "tmp" in var:
                        p += getTypeString() + " " + var + ";\n"
                p += "#pragma omp critical\n"
                p += "// Low: {}, High: {}, {}\n".format(low, high, self.definedCriticalVariables)
                p += "// Shared: {}\n".format(self.dataSharingAttribs.getSharedVars())
                p += "// Private: {}\n".format(self.dataSharingAttribs.getPrivateVars())
                p += "// FirstPrivate: {}\n".format(self.dataSharingAttribs.getFirstPrivateVars())
                p += "{\n"

            ret.append(p + l.printCode())
                                            
            if high != None and high == idx and self.id  == inCriticalSection:
                ret.append("\n}")
                inCriticalSection = None 


        if prevLineCritical:
            ret.append("}\n")
                
        return "\n".join(ret)
    
    def setDataSharingAttributes(self, dataSharingAttributes: DataSharingAttributes):
        self.dataSharingAttribs = dataSharingAttributes


# Types of binary operations
class BooleanExpressionType(Enum):
    eq = 0 # equal than
    lt = 1 # less than
    gt = 2 # greater than
    geq = 3 # greater or equal than
    leq = 4 # less or equal than

class BooleanExpression(Node):
    def __init__(self, code="==", left=None, right=None):
        #self.idGen = idGen
        op = random.choice(list(BooleanExpressionType))
        if op == BooleanExpressionType.eq:
            self.code = " == "
        elif op == BooleanExpressionType.lt:
            self.code = " < "
        elif op == BooleanExpressionType.gt:
            self.code = " > "
        elif op == BooleanExpressionType.geq:
            self.code = " >= "
        elif op == BooleanExpressionType.leq:
            self.code = " <= "

        self.left = "comp"
        self.right = Expression()
        
        self.usedVars = {"comp"}.union(self.right.usedVars)

    def printCode(self) -> str:
        return self.left + self.code + self.right.printCode(False)

class ForLoopCondition(Node):
    def __init__(self, code="", left=None, right=None):
        self.variableBoundOn = id_generator.IdGenerator.get().generateIntID()
        self.code = "int i=0; i < " + self.variableBoundOn + "; ++i"
        self.usedVars = {self.variableBoundOn}
    
    def printCode(self) -> str:
        return self.code
    
class IfConditionBlock(Node):
    def __init__(self, level=1, code=None, left=None, right=None, recursive=True, isParallel=False,
                 dataSharingAttribs=DataSharingAttributes()):
        self.level = level
        self.indentation = ''
        self.indentation += '  ' * self.level
        self.rec = recursive
        self.isParallel = isParallel 
        self.dataSharingAttribs = dataSharingAttribs
        
        # Generate code of the boolean expresion (default)
        self.code = BooleanExpression()
        self.usedVars = self.code.usedVars 
        
        # Generate code inside the block
        self.left = left
        self.right = "break;"
        
        if self.left == None:
            self.left = OperationsBlock(recursive=self.rec, isParallel=self.isParallel,
                                        dataSharingAttribs=self.dataSharingAttribs)
            
        self.usedVars = self.usedVars.union(self.left.usedVars)

    def printCode(self) -> str:
        calledNodes.append("IfConditionBlock")
        t = "if (" + self.code.printCode() + ") {\n"
        #t = t + self.indentation + self.left.printCode() + "\n"
        for line in self.left.printCode().splitlines():
            t += self.indentation + line + "\n"
        t += "}\n"
        return t

    def setContent(self, c):
        # Resetting used variables to just the boolean expression.
        print("Before: ", self.usedVars)
        self.usedVars = self.code.usedVars

        self.left = c
        if not isinstance(c, str):
            self.usedVars = self.usedVars.union(c.usedVars)
        
        print("After: ", self.usedVars)

class ForLoopBlock(Node):
    def __init__(self, level=1, code=None, left=None, right=None, recursive=True, isParallel=False,
                 dataSharingAttribs=DataSharingAttributes()):
        self.level = level
        self.indentation = ''
        self.indentation += '  ' * self.level
        self.rec = recursive
        self.isParallel = isParallel
        self.dataSharingAttribs = dataSharingAttribs

        # Generate code of the loop condition
        self.code = ForLoopCondition()
        self.usedVars = self.code.usedVars 
        #self.left = OperationsBlock()
        self.left = left
        self.right = None
        
        if self.isParallel and left != None:
            self.left.isParallel = True

        if self.left == None:
            self.left = OperationsBlock(level=self.level+1, inLoop=True, recursive=self.rec, isParallel=self.isParallel,
                                        dataSharingAttribs=self.dataSharingAttribs)

        self.usedVars = self.usedVars.union(self.left.usedVars)
        self.generateDataSharingAttribs()
        
        if isinstance(self.left, OperationsBlock):
            self.left.setDataSharingAttributes(self.dataSharingAttribs)
            
    def generateDataSharingAttribs(self):
        # We can't just simply pass the IDs from this list for two reasons:
        #   1. This doesn't include the 'comp' variable
        #   2. This list WILL include the variable the for-loop is bound on
        #     * We don't want to do this with OpenMP's behavior.
        vars_list = [x for x in id_generator.IdGenerator.get().getVarsList().keys()]
        try:
            vars_list.remove(self.code.variableBoundOn)
            #print("Variable {} i sin list and bound".format(self.code.variableBoundOn))
        except ValueError:
            print("Variable {} not in list but bounding".format(self.code.variableBoundOn))
        vars_list.append("comp")
        attribs = randomListChunk(vars_list, n=3)
        self.dataSharingAttribs = DataSharingAttributes(
            sharedVars=attribs[0],
            privateVars=attribs[1],
            firstPrivateVars=attribs[2]
        )
        
    def setDataSharingAttributes(self, sharedVars=[], privateVars=[], firstPrivateVars=[]):
        self.dataSharingAttribs = DataSharingAttributes(
            sharedVars=sharedVars,
            privateVars=privateVars,
            firstPrivateVars=firstPrivateVars
        )
        
    def printBody(self) -> str:
        t = "for (" + self.code.printCode() + ") {\n"
        #t = t + self.indentation + self.left.printCode() + "\n"
        for line in self.left.printCode().splitlines():
            t += self.indentation + line + "\n"
        t += "}\n"
        return t

    def printDataSharingAttributes(self) -> str:
        sv = ""
        if len(self.dataSharingAttribs.getSharedVars()) >= 1:
            sv = "shared(" + ", ".join(self.dataSharingAttribs.getSharedVars()) + ")"
        pv = ""
        if len(self.dataSharingAttribs.getPrivateVars()) >= 1:
            pv = "private(" + ", ".join(self.dataSharingAttribs.getPrivateVars()) + ")"
        fpv = ""
        if len(self.dataSharingAttribs.getFirstPrivateVars()) >= 1:
            fpv = "firstprivate(" + ", ".join(self.dataSharingAttribs.getFirstPrivateVars()) + ")"

        t = "#pragma omp parallel default(shared) {} {} {} \n".format(sv, pv, fpv)
        t += "{\n"
        # Temporal variables used for intermmediate computations can be defined in
        # critical sections, but this reduces their scope to only those critical blocks. Instead,
        # to make this mimic more "real-world" scenarios, we can declare the variables
        # at the beginning of the parallel block so that they are available to all subsequent blocks.
        #for var in self.usedVars:
        #       if "tmp" in var:
        #           t += getTypeString() + " " + var + ";\n"

        # Since the behavior of private variable values is undefined initially, we
        # will immediately just assign the private variables to 0, or if they are a
        # pointer, we will have to re-alloc and then automatically assign 0 to each
        # indices.
        init_private_pointer = ""
        init_private_int = ""
        init_private_real = ""
        var_type = id_generator.IdGenerator.get().getVarsList()
        for var in self.dataSharingAttribs.getPrivateVars():
            type = var_type[var]
            if isTypeInt(type):
               init_private_int += "{} = 0;\n".format(var)
            elif isTypeReal(type):
               init_private_real += "{} = 0.0;\n".format(var)
            elif isTypeRealPointer(type): 
               init_private_pointer += "{} = initDynamicArray(0, {});\n".format(var, cfg.ARRAY_SIZE)
#        if len(self.dataSharingAttribs.getPrivateVars()) >= 1:
#            t += " = ".join(self.dataSharingAttribs.getSharedVars()) + " = 0;\n" 
        t += init_private_pointer + init_private_int + init_private_real + "#pragma omp for\n"
        return t

    def printCode(self) -> str:
        calledNodes.append("ForLoopBlock")
        global parallel_region_generated

        c = ""
        # We mark this here so that any inner-loops don't use
        # incorrectly use parallel for-loop expressions.
        if self.isParallel and not parallel_region_generated:
            parallel_region_generated = True 
            c = self.printBody()
            c = self.printDataSharingAttributes() + c + "\n}\n"
            parallel_region_generated = False
        else:
            c = self.printBody()
        
        return c 

    def setContent(self, c):
        # Resetting used variables to just the boolean expression.
        self.usedVars = self.code.usedVars

        self.left = c
        if not isinstance(c, str):
            self.usedVars = self.usedVars.union(c.usedVars)

class CodeBlock(Enum):
    expression = 1
    if_codition = 2
    for_loop = 3
#    while_loop = 4

class FunctionCall(Node):
    #global MAX_NESTING_LEVELS
    
    def __init__(self, code=None, left=None, right=None, parameterVarNames=[]):
        #self.device = device
        self.code = None
        self.left = None
        self.right = "}\n"
        #self.idGen = idGen
        self.codeCache = None # If the code was printed will be saved here    
        self.parameterVarNames = parameterVarNames
    
        # Sample the blocks and levels of the function
        levels = random.randrange(1, cfg.MAX_NESTING_LEVELS+1)
        #levels = 2
        #print("levels: {}".format(levels))
        lastBlock = None
        
        blocks = []
        while(levels >= 1):
            b = random.choice(list(CodeBlock))
            blocks.append(b)
            levels = levels-1
        
        parallel_for_blocks = []
        #blocks = [CodeBlock.if_codition, CodeBlock.if_codition, CodeBlock.expression]
        for i in range(len(blocks)):
            b = blocks[i]
            if b == CodeBlock.expression:
                c = OperationsBlock()
                if lastBlock != None:
                    lastBlock.setContent(c)
                
                # set the body of the function
                if self.left == None:
                    self.left = c
                
                parallel_for_blocks.append(c)
                blocks = blocks[:i+1]
                break
                              
            elif b == CodeBlock.if_codition:
                c = IfConditionBlock(i+1)
                if lastBlock != None:
                    lastBlock.setContent(c)
                lastBlock = c
            
                # set the body of the function
                if self.left == None:
                    self.left = c
                parallel_for_blocks.append(lastBlock)
                              
            elif b == CodeBlock.for_loop:
                # We want to set data sharing attributes here for the first time.
                # Subsequent blocks under parallel sections will maintain the same
                # data sharing attributes mapping as it is parameterized.
                if cfg.PARALLEL_PROG:
                    if lucky():
                        c = ForLoopBlock(i+1, isParallel=True)
                    else:
                        c = ForLoopBlock(i+1)
                else:
                    c = ForLoopBlock(i+1)
                
                if lastBlock != None:
                    lastBlock.setContent(c)
                lastBlock = c
                
                # set the body of the function
                if self.left == None:
                    self.left = c
                parallel_for_blocks.append(lastBlock)
                
            else:
                parallel_for_blocks.append(None)
                
#        print(len(parallel_for_blocks))
#        print(len(blocks))
#        for i in range(len(blocks)):
#            b = blocks[i]
#            if b == CodeBlock.for_loop:
#                if parallel_for_blocks[i].isParallel:
#                    attribs = randomListChunk(["comp"].extend(id_generator.IdGenerator.get().getVarsList().keys()), n=3)
#                    parallel_for_blocks[i].setDataSharingAttributes(
#                        sharedVars=attribs[0],
#                        privateVars=attribs[1],
#                        firstPrivateVars=attribs[2]
#                    )
                    
                    
    def getParameters(self):
        return self.parameterVarNames
                              
    def printHeader(self):
        h = ""
        #if self.device == True:
        #    h = h + "__global__ "
        h = h + "void compute("
        h = h + getTypeString() + " comp"
        if len(id_generator.IdGenerator.get().printAllVars()) > 0:
            h = h + ", "
        h = h + ",".join(id_generator.IdGenerator.get().printAllVars())
        h = h + ") {\n"
        if cfg.USE_TIMERS:
            h = h + self.writeTimeBegin()
        return h

    def writePrintStatement(self):
        ret = ""
        if cfg.USE_TIMERS:
            ret += '\n   printf("%.17g ", comp);\n'
            ret += self.writeTimeEnd()
        else:
            ret = '\n   printf("%.17g\\n", comp);\n'
        return ret
        #return '\n   printf("%.17g\\n", comp);\n'
    
    def writeTimeBegin(self):
        return '   std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();\n'
    
    def writeTimeEnd(self):
        ret = '   std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();\n'
        ret += '   std::cout << "time:" << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;\n'
        return ret
        
    def printCode(self) -> str:
        if self.codeCache != None:
            return self.codeCache

        c = self.left.printCode()
        c = self.printHeader() + c
        c = c + self.writePrintStatement()
        c = c + "\n}"
        self.codeCache = c
        return c

class Program():
    def __init__(self):
        id_generator.IdGenerator.get().clear()
        self.func = FunctionCall()
        
    def printInputVariables(self):
        ret = ""
        vars = id_generator.IdGenerator.get().getVarsList()
        
        ret = ret + "  " + getTypeString() + " " + "tmp_1 = atof(argv[1]);\n"
        idNum = 2
        for k in vars.keys():
            type = vars[k]
            if (isTypeReal(type)):
                ret = ret + "  " + getTypeString() + " " + "tmp_" + str(idNum)
                ret = ret + " = atof(argv[" + str(idNum) + "]);\n"
            elif (isTypeInt(type)):
                ret = ret + "  int " + "tmp_" + str(idNum)
                ret = ret + " = atoi(argv[" + str(idNum) + "]);\n"
            elif (isTypeRealPointer(type)):
                ret = ret + "  "+type+" " + "tmp_" + str(idNum)
                # Kind of hacky to just assume the size is based off the previous input integer, but that's how
                # the current code-base is structured.
                #ret = ret + " = initPointer( atof(argv[" + str(idNum) + "]), atoi(argv[" + str(idNum-1) + "]) );\n"
                ret = ret + " = initDynamicArray( atof(argv[" + str(idNum) + "]), {});\n".format(cfg.ARRAY_SIZE)

            idNum = idNum + 1

        ret = ret + "\n"
        return ret
    
    def printFunctionParameters(self):
        vars = []
        for k in range(len(id_generator.IdGenerator.get().getVarsList()) + 1):
            vars.append("tmp_" + str(k+1))
        return ",".join(vars)

    def printPointerInitFunction(self):
        ret = "\n"+getTypeString()+"* initDynamicArray("+getTypeString()+" v, int arraySize) {\n"
        ret = ret + "  "+getTypeString()+" *ret = "
        ret = ret + "("+getTypeString()+"*) malloc(sizeof("+getTypeString()+")* arraySize);\n"
        ret = ret + "  for(int i=0; i < arraySize; ++i)\n"
        ret = ret + "    ret[i] = v;\n"
        ret = ret + "  return ret;\n"
        ret = ret + "}"
        return ret
    
    def printStaticInitFunction(self):
        ret = "\n"+getTypeString()+"* initStaticArray("+getTypeString()+" v, int arraySize) {\n"
        ret = ret + "  "+getTypeString()+" *ret = "
        ret = ret + "("+getTypeString()+"*) malloc(sizeof("+getTypeString()+")* arraySize);\n"
        ret = ret + "  for(int i=0; i < arraySize; ++i)\n"
        ret = ret + "    ret[i] = v;\n"
        ret = ret + "  return ret;\n"
        ret = ret + "}"
        return ret

    def printHeader(self):
        h = "\n/* This is a automatically generated test. Do not modify */\n\n"
        h = h + "#include <stdio.h>\n"
        h = h + "#include <stdlib.h>\n"
        if cfg.USE_TIMERS:
            h = h + "#include <iostream>\n"
            h = h + "#include <chrono>\n"
        h = h + "#include <math.h>\n\n"

        # Function declaration before compute function so we can
        # call init function when arrays are private in parallel
        # blocks.
        h += getTypeString()+"* initDynamicArray("+getTypeString()+" v, int arraySize);\n"
        
        if self.device == True:
            h = h + "__global__\n"
        h = h + self.func.printCode()
        h = h + "\n" + self.printPointerInitFunction()
        h = h + "\n\nint main(int argc, char** argv) {\n"
        h = h + "/* Program variables */\n\n"
        h = h + self.printInputVariables()
        return h

    def printCode(self, device=False) -> (str,str):
        self.device = device
        c = self.printHeader()
        # call the function
        if self.device == False:
            c = c + "  compute(" + self.printFunctionParameters() + ");\n"
        else: # here we call a device kernel
            c = c + "  compute<<<1,1>>>(" + self.printFunctionParameters() + ");\n"
            c = c + "  cudaDeviceSynchronize();\n"

        # finalize main function
        c = c + "\n  return 0;\n"
        c = c + "}\n"
        allTypes = ",".join(id_generator.IdGenerator.get().printAllTypes())
        return (c, allTypes)
        
    def compileProgram(self, device=False):
        (code, allTypes) = self.printCode(device)
        if self.device == False:
            fileName = 'tmp.c'
        else:
            fileName = 'tmp.cu'

        fd =open(fileName, 'w')
        fd.write(code)
        fd.close()
    
        print("Compiling: " + fileName)
        try:
            if self.device == False:
                cmd = "clang -std=c99 -o " + fileName + ".exe " + fileName
            else: # compile for device case
                cmd = "nvcc -o " + fileName + ".exe " + fileName

            print(cmd)
            out = subprocess.check_output(cmd, shell=True)                    
        except subprocess.CalledProcessError as outexc:                                                                                                   
            print ("Error at compile time:", outexc.returncode, outexc.output)

    def runProgram(self):
        print("Running...")
        input = self.getInput()
        try:
            if self.device == False:
                cmd = "./tmp.c.exe " + input
            else:
                cmd = "./tmp.cu.exe " + input

            out = subprocess.check_output(cmd, shell=True) 
            res = out.decode('ascii')[:-1]
            print(cmd)
            print(res)
        except subprocess.CalledProcessError as outexc:                                                                                                   
            print ("Error at runtime:", outexc.returncode, outexc.output)

    def getInput(self):
        allTypes = ",".join(id_generator.IdGenerator.get().printAllTypes())
        print("ALL TYPES", allTypes)
        #inGen = gen_inputs.InputGenerator()
        input = gen_inputs.InputGenerator.genInput() + " "
        typeList = allTypes.split(",")
        for type in typeList:
            if isTypeReal(type) or isTypeRealPointer(type):
                input = input + gen_inputs.InputGenerator.genInput() + " "
            elif isTypeInt(type):
                input = input + "5 "
        return input

if __name__ == "__main__":

    #p = Program()
    #(c, allTypes) = p.printCode(True)
    #print(p.printCode()[0])
    #print(p.printCode(True)[0])
    
    #print(calledNodes)
    
    # Compile and run program 
    #p.compileProgram()
    #p.runProgram()

    #o = OperationsBlock(inLoop=True)
    #print(o.printCode())
    
    # ---- Class Testing ----
    fb = ForLoopBlock()
    print(fb.printCode())
    #ep = Expression()
    #print(ep.printCode())
    print(calledNodes)