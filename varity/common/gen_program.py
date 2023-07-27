
import gen_inputs
import id_generator
import gen_inputs
import cfg

from enum import Enum
import random
import subprocess

from random_functions import lucky, veryLucky, generateMathExpression
from type_checking import getTypeString, isTypeReal, isTypeRealPointer, isTypeInt

# ================= Global State ===================
parallel_region_generated = False
calledNodes = [] # stack of nodes created (ordered)
# ==================================================

# Basic node in a tree
class Node:
    def __init__(self, code, left=None, right=None):
        self.code = code
        self.left  = left
        self.right = right

    def __str__(self):
        return str(self.code)

    def printCode(self) -> str:
        return "{};\n"

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
    def __init__(self, code="=", left=None, right=None, varToBeUsed=None, parallel=True):
        import gen_math_exp
        self.left  = left
        self.right = right
        self.varToBeUsed = varToBeUsed
        self.parallel = parallel
        
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
            else:
                op = BinaryOperation()
                op.generate()
            
            if lastOp != None:
                lastOp.right = op
            else:
                self.rootNode = op
            lastOp = op
            size = size - 1

            # If we have a math expression, this is a terminator for the tree
            if mathExpTerminator == True:
                break

    def total(self, n):
        import gen_math_exp
        if n == None:
            if lucky():
                n = gen_inputs.InputGenerator.genInput()
            else:
                n = id_generator.IdGenerator.get().generateRealID()
            return n
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
        
        if assignment == True:
            p = ""
            if self.parallel:
                p = "#pragma omp critical\n"
            return p + "comp " + self.code + " " + t + ";"
        else:
            return t

class VariableDefinition(Node):
    def __init__(self, code=" = ", left=None, right=None, isPointer=False, parallel=True):
        self.code = code
        self.right = right
        self.isPointer = isPointer
        self.parallel = parallel
        
        if isPointer == True:
            self.left  = id_generator.IdGenerator.get().generateRealID(True) + "[i]"
        else:
            self.left  = getTypeString() + " " + id_generator.IdGenerator.get().generateTempRealID()
        
        if lucky(): # constant definition
            self.right = gen_inputs.InputGenerator.genInput()
        else:
            self.right = Expression()

    def getVarName(self):
        if self.isPointer == False:
            return self.left.split(" ")[1]
        else:
            return self.left
   
    def printCode(self) -> str:
        calledNodes.append("VariableDefinition")
        if isinstance(self.right, str):
            c = self.right
        else:
            c = self.right.printCode(False)
        
        p = ""
        if self.parallel and self.isPointer:
            p = "#pragma omp critical\n"
            
        return p + self.left + self.code + c + ";"

# A non-recursive block has only expressions (it does not have if-blocks or loop-blocks)
class OperationsBlock(Node):
    def __init__(self, code="", left=None, right=None, inLoop=False, recursive=True, parallel=False):
        self.code = code
        self.left  = left
        self.right = right
        
        # Defines the number of lines that the block will have
        lines = random.randrange(1, cfg.MAX_LINES_IN_BLOCK+1)
        assert lines > 0

        # In the block, we either have definitions of new variables or 
        # assigments to comp. The last line of the block will always be 
        # an assigment from an expression:
        #    comp = ...
        if lines == 1:
            self.left = [Expression(parallel=parallel)]
        else:
            i = 1
            varsToBeUsed = []
            l = []
            while(i <= lines):
                    
                if lucky() or i==lines: # expression with assigment
                    c = None
                    if len(varsToBeUsed) > 0:
                        c = Expression("=", None, None, varsToBeUsed[:],parallel)
                        varsToBeUsed.clear()
                    else:
                        c = Expression(parallel=parallel)
                    l.append(c)
                    if i==lines:
                        break
                else:
                    if inLoop==True and lucky():
                        v = VariableDefinition(isPointer=True, parallel=parallel)
                    else:
                        v = VariableDefinition(parallel=parallel)
                    l.append(v)
                    varsToBeUsed.append(v.getVarName())
                i = i+1
            self.left = l

        # An operations block can also have if-conditions and loop blocks
        if recursive:
            nBlocks = random.randrange(0, cfg.MAX_SAME_LEVEL_BLOCKS+1)
            #nBlocks = 2
            for k in range(nBlocks):
                if lucky():
                    b = IfConditionBlock(recursive=False, parallel=parallel)
                else:
                    if cfg.PARALLEL_PROG:
                        b = OMPParallelForLoopBlock(recursive=False)
                    else:
                        b = ForLoopBlock(recursive=False)
                self.left.append(b)
                    
    def printCode(self) -> str:
        calledNodes.append("OperationsBlock")
        ret = []
        for l in self.left:
            ret.append( l.printCode() )
        return "\n".join(ret)

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

    def printCode(self) -> str:
        return self.left + self.code + self.right.printCode(False)

class FoorLoopCondition(Node):
    def __init__(self, code="", left=None, right=None):
        self.code = "int i=0; i < " + id_generator.IdGenerator.get().generateIntID() + "; ++i"
    
    def printCode(self) -> str:
        return self.code
    
class IfConditionBlock(Node):
    def __init__(self, level=1, code=None, left=None, right=None, recursive=True, parallel=False):
        self.level = level
        self.identation = ''
        self.identation += '  ' * self.level
        self.rec = recursive
        self.parallel = parallel
        
        # Generate code of the boolean expresion (default)
        self.code = BooleanExpression()
        
        # Generate code inside the block
        self.left = left
        self.right = "break;"

    def printCode(self) -> str:
        calledNodes.append("IfConditionBlock")
        t = "if (" + self.code.printCode() + ") {\n"
        if self.left == None:
            self.left = OperationsBlock(recursive=self.rec, parallel=self.parallel)
        t = t + self.identation + self.left.printCode() + "\n"
        t = t + "}"
        return t

    def setContent(self, c):
        self.left = c

class ForLoopBlock(Node):
    def __init__(self, level=1, code=None, left=None, right=None, recursive=True):
        self.level = level
        self.identation = ''
        self.identation += '  ' * self.level
        self.rec = recursive

        # Generate code of the loop condition
        self.code = FoorLoopCondition()
        #self.left = OperationsBlock()
        self.left = left
        self.right = None

    def printCode(self) -> str:
        calledNodes.append("ForLoopBlock")
        return printMainBody()

    def printMainBody(self) -> str:
        t = "for (" + self.code.printCode() + ") {\n"
        if self.left == None:
            self.left = OperationsBlock(inLoop=True, recursive=self.rec, parallel=cfg.PARALLEL_PROG)
        t = t + self.identation + self.left.printCode() + "\n"
        t = t + "}"
        return t

    def setContent(self, c):
        self.left = c

class OMPParallelForLoopBlock(ForLoopBlock):
    def __init__(self, level=1, code=None, left=None, right=None, recursive=True,
                 varsToBeUsed={"shared": [], "private": [], "firstprivate": []}):
        super().__init__(level=level, code=code, left=left, right=right, recursive=recursive)
        self.varsToBeUsed = varsToBeUsed
        if left:
            self.left.parallel = True

    def printCode(self) -> str:
        calledNodes.append("OMPParallelForLoopBlock")
        global parallel_region_generated
        
        t = ""
        if not parallel_region_generated:
            t += "#pragma omp parallel default(shared)\n"
            parallel_region_generated = True
            
        return t + super().printMainBody()

class CodeBlock(Enum):
    expression = 1
    if_codition = 2
    for_loop = 3
#    while_loop = 4

class FunctionCall(Node):
    #global MAX_NESTING_LEVELS
    
    def __init__(self, code=None, left=None, right=None):
        #self.device = device
        self.code = None
        self.left = None
        self.right = "}\n"
        #self.idGen = idGen
        self.codeCache = None # If the code was printed will be saved here    
    
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
                
                break
                              
            elif b == CodeBlock.if_codition:
                c = IfConditionBlock(i+1)
                if lastBlock != None:
                    lastBlock.setContent(c)
                lastBlock = c
            
                # set the body of the function
                if self.left == None:
                    self.left = c
                              
            elif b == CodeBlock.for_loop:
                if cfg.PARALLEL_PROG:
                    c = OMPParallelForLoopBlock(i + 1)
                else:
                    c = ForLoopBlock(i+1)
                
                if lastBlock != None:
                    lastBlock.setContent(c)
                lastBlock = c
                
                # set the body of the function
                if self.left == None:
                    self.left = c
                              
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
                ret = ret + " = initPointer( atof(argv[" + str(idNum) + "]) );\n"

            idNum = idNum + 1

        ret = ret + "\n"
        return ret
    
    def printFunctionParameters(self):
        vars = []
        for k in range(len(id_generator.IdGenerator.get().getVarsList()) + 1):
            vars.append("tmp_" + str(k+1))
        return ",".join(vars)

    def printPointerInitFunction(self):
        ret = "\n"+getTypeString()+"* initPointer("+getTypeString()+" v) {\n"
        ret = ret + "  "+getTypeString()+" *ret = "
        ret = ret + "("+getTypeString()+"*) malloc(sizeof("+getTypeString()+")*"+ str(cfg.ARRAY_SIZE) +");\n"
        ret = ret + "  for(int i=0; i < "+ str(cfg.ARRAY_SIZE) +"; ++i)\n"
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
    

    

