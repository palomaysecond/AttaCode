"""
@author: xuty
@contact:xuty1218@163.com
@version: 1.0.0
@file: GetAST.py
@time: 2025/5/26 11:00
"""
from tree_sitter import Parser
from tree_sitter import Tree
from tree_sitter import Language

Language.build_library(
    'build/my-languages.so',
    [
        'vendor/tree-sitter-python',
        'vendor/tree-sitter-java',
        'vendor/tree-sitter-c',
        'vendor/tree-sitter-cpp',
    ]
)

JAVA_LANGUAGE=Language('build/my-languages.so','java')
PYTHON_LANGUAGE=Language('build/my-languages.so','python')
C_LANGUAGE = Language('build/my-languages.so', 'c')
CPP_LANGUAGE = Language('build/my-languages.so', 'cpp')


parser=Parser()


def generateASt(code,language):  
    if language=='java':
        parser.set_language(JAVA_LANGUAGE)  
    elif language=='python':
        parser.set_language(PYTHON_LANGUAGE)
    elif language=='c':
        parser.set_language(C_LANGUAGE)
    elif language=='cpp':
        parser.set_language(CPP_LANGUAGE)
    else:
        print('--wrong langauge--')
        return 0
    tree=parser.parse(bytes(code,encoding='utf-8'))  
    root_node=tree.root_node 
    return root_node
