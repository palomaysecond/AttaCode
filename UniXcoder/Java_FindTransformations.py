import numpy as np
import json
import argparse
from GetAST import generateASt
from ImportanceAnalyze_unix import load_preprocessed_data


def build_token_map(source_code: str, token_list: list, scores: np.ndarray):
    # Precompute line offsets
    lines = source_code.splitlines()
    line_starts = []
    offset = 0
    for line in lines:
        line_starts.append(offset)
        offset += len(line) + 1  # +1 for '\n'

    # Utility to find next occurrence of token from index
    def find_token_position(src, start_idx, token):
        max_len = len(src)
        token_len = len(token)
        while start_idx < max_len:
            if src[start_idx:start_idx + token_len] == token:
                return start_idx
            start_idx += 1
        return None

    # Start constructing token_map
    token_map = [[] for _ in lines]
    src = source_code
    src_idx = 0
    score_len = len(scores)

    for j, token in enumerate(token_list):
        raw_token = token.lstrip('_')  # ignore leading underscore for matching
        if raw_token == '' or raw_token in {'<s>', '</s>'}:
            continue  # skip special or empty tokens

        score = float(scores[j]) if j < score_len else 0.0
        match_idx = find_token_position(src, src_idx, raw_token)
        if match_idx is None:
            continue  # skip if not found

        # Determine which line the match occurs in
        line_num = 0
        while line_num + 1 < len(line_starts) and match_idx >= line_starts[line_num + 1]:
            line_num += 1

        col_start = match_idx - line_starts[line_num]
        col_end = col_start + len(raw_token)
        # if token.startswith("_"):
        #     token.replace('_', '')
        token_map[line_num].append((raw_token, line_num, col_start, col_end, score))

        # Move the cursor forward
        src_idx = match_idx + len(raw_token)

    return token_map


def walk_tree(node):
    yield node
    for child in node.children:
        yield from walk_tree(child)


def AssessForToWhileConversion(node, source_code):

    if node.type in ['for_statement', 'enhanced_for_statement']:
        return (True, node.start_byte , node.end_byte, node.start_point, node.end_point)
    return False


def AssessWhileToForRefactoring(node, source_code):

    if node.type == 'while_statement':
        return (True, node.start_byte , node.end_byte, node.start_point, node.end_point)
    return False


def AssessDoWhileToWhileConversion(node, source_code):

    if node.type == 'do_statement':
        return (True, node.start_byte , node.end_byte, node.start_point, node.end_point)
    return False


def AssessInlineLoopDeclaration(node, source_code):

    if node.type != 'local_variable_declaration':
        return False
    declared_vars = set()
    for declarator_node in node.children:
        if declarator_node.type == 'variable_declarator':
            value_node = declarator_node.child_by_field_name('value')
            if value_node is None:
                name_node = declarator_node.child_by_field_name('name')
                if name_node and name_node.type == 'identifier':
                    var_name = source_code[name_node.start_byte:name_node.end_byte]
                    declared_vars.add(var_name)
    
    if not declared_vars:
        return False
    
    for_node = node.next_named_sibling
    if for_node is None or for_node.type != 'for_statement':
        return False

    init_node = for_node.child_by_field_name('init')
    if init_node and init_node.type == 'expression_statement':
        if init_node.named_child_count > 0 and init_node.named_children[0].type == 'assignment_expression':
            assignment_expr = init_node.named_children[0]
            loop_var_node = assignment_expr.child_by_field_name('left')
            if loop_var_node and loop_var_node.type == 'identifier':
                loop_var_name = source_code[loop_var_node.start_byte:loop_var_node.end_byte]
                
                if loop_var_name in declared_vars:
                    start_byte = node.start_byte
                    start_point = node.start_point
                    body_node = for_node.child_by_field_name('body')
                    if body_node:
                        end_byte = body_node.start_byte
                        end_point = body_node.start_point
                    else:
                        end_byte = for_node.end_byte
                        end_point = for_node.end_point
                    return (True, start_byte, end_byte, start_point, end_point)

    return False

def AssessExtractLoopDeclaration(node, source_code):

    if node.type == 'for_statement':
        init_node = node.child_by_field_name('init')

        if init_node and init_node.type == 'local_variable_declaration':

            body_node = node.child_by_field_name('body')
            if body_node:
                end_byte = body_node.start_byte
                r_paren_node = node.child(node.child_count - 2)
                if r_paren_node and r_paren_node.type == ')':
                    end_byte = r_paren_node.end_byte
            return (True, node.start_byte, end_byte, node.start_point, (body_node.start_point[0], body_node.start_point[1]))
    return False


def AssessIfElseBranchSwap(node, source_code):

    if node.type == 'if_statement':
        consequence_node = node.child_by_field_name('consequence')
        alternative_node = node.child_by_field_name('alternative')
        if not (consequence_node and alternative_node):
            return False

        condition_node = node.child_by_field_name('condition')
        if condition_node and condition_node.named_child_count > 0:
            actual_condition = condition_node.named_children[0]
            if actual_condition.type == 'binary_expression':
                for child in actual_condition.children:
                    if child.type in ['==', '!=']:
                        return (True, condition_node.start_byte, condition_node.end_byte, condition_node.start_point, condition_node.end_point)
    return False


def AssessElseIfToNestedIf(node, source_code):

    if node.type != 'if_statement':
        return False
    consequence_node = node.child_by_field_name('consequence')
    if not (consequence_node and consequence_node.type == 'block'):
        return False
    parent = node.parent
    if not parent or parent.type != 'if_statement':
        return False
    alternative_node = parent.child_by_field_name('alternative')
    if not (alternative_node and alternative_node.id == node.id):
        return False
    node_index = -1
    for i, child in enumerate(parent.children):
        if child.id == node.id:
            node_index = i
            break
    if node_index > 0:
        prev_sibling = parent.children[node_index - 1]
        if prev_sibling.type == 'else':
            return (True, prev_sibling.start_byte, node.end_byte, prev_sibling.start_point, node.end_point)
    return False


def AssessNestedIfToElseIf(node, source_code):

    if node.type != 'if_statement':
        return False
    consequence_node = node.child_by_field_name('consequence')
    if not (consequence_node and consequence_node.type == 'block'):
        return False
    parent_block = node.parent
    if not parent_block or parent_block.type != 'block' or parent_block.named_child_count != 1:
        return False
    grandparent_if = parent_block.parent
    if not grandparent_if or grandparent_if.type != 'if_statement':
        return False
    alternative_node = grandparent_if.child_by_field_name('alternative')
    if not (alternative_node and alternative_node.id == parent_block.id):
        return False
    node_index = -1
    for i, child in enumerate(grandparent_if.children):
        if child.id == parent_block.id:
            node_index = i
            break
    if node_index > 0:
        prev_sibling = grandparent_if.children[node_index - 1]
        if prev_sibling.type == 'else':
            return (True, prev_sibling.start_byte, parent_block.end_byte, prev_sibling.start_point, parent_block.end_point)
    return False

def AssessUnwrapRedundantBlock(node, source_code):

    if node.type == 'if_statement':
        consequence = node.child_by_field_name('consequence')
        if consequence and consequence.type == 'block':
            # 范围就是这个 block
            return (True, consequence.start_byte, consequence.end_byte, consequence.start_point, consequence.end_point)
    return False

def AssessWrapStatementInBlock(node, source_code):

    if node.type == 'if_statement':
        consequence = node.child_by_field_name('consequence')
        if consequence and consequence.type != 'block':
            # 范围就是这个无{}块状的执行体
            return (True, consequence.start_byte, consequence.end_byte, consequence.start_point, consequence.end_point)
    return False


def AssessSplitCompoundCondition(node, source_code):

    if node.type == 'if_statement':
        consequence_node = node.child_by_field_name('consequence')
        if not (consequence_node and consequence_node.type == 'block'):
            return False
        condition_node = node.child_by_field_name('condition')
        if condition_node and condition_node.named_child_count > 0:
            actual_condition = condition_node.named_children[0]
            if actual_condition.type == 'binary_expression':
                for child in actual_condition.children:
                    if child.type in ['&&', '||']:
                        return (True, condition_node.start_byte, condition_node.end_byte, condition_node.start_point, condition_node.end_point)
    return False

def AssessAddRedundantStatement(node, source_code):

    if node.type == 'local_variable_declaration':
        return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)
    return False

def AssessWrapWithConstantCondition(node, source_code):

    if node.type != 'expression_statement':
        return False

    if not (node.named_child_count > 0 and node.named_children[0].type == 'assignment_expression'):
        return False

    if node.parent and node.parent.type == 'block':
        return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)

    return False
    

def AssessExtractSubexpression(node, source_code):

    if not (node.type == 'expression_statement' and node.parent and node.parent.type == 'block'):
        return False

    if not (node.named_child_count > 0 and node.named_children[0].type == 'assignment_expression'):
        return False
    
    assignment_node = node.named_children[0]

    rhs_node = assignment_node.child_by_field_name('right')
    if not (rhs_node and rhs_node.type == 'binary_expression'):
        return False

    left_operand = rhs_node.child_by_field_name('left')
    right_operand = rhs_node.child_by_field_name('right')

    if (left_operand and left_operand.type in ['binary_expression', 'method_invocation']) or \
       (right_operand and right_operand.type in ['binary_expression', 'method_invocation']):
        return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)

    return False


def AssessSwitchToIfElse(node, source_code):

    if node.type in ['switch_expression', 'switch_statement']:
        return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)
    return False


def AssessReturnViaTempVariable(node, source_code):

    if node.type == 'return_statement':
        if node.named_child_count > 0:
            return_value_node = node.named_children[0]
            literal_types = {
                'decimal_integer_literal', 'hex_integer_literal', 'octal_integer_literal',
                'binary_integer_literal', 'decimal_floating_point_literal', 'hex_floating_point_literal',
                'string_literal', 'character_literal', 'boolean_literal', 'null_literal',
            }
            if return_value_node.type in literal_types:
                return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)
    return False


def AssessNegateWithReversedOperator(node, source_code):

    if node.type == 'binary_expression':
        comparison_operators = {'<', '>', '<=', '>=', '==', '!='}
        for child in node.children:
            if child.type in comparison_operators:
                return (True, node.start_byte , node.end_byte, node.start_point, node.end_point)
    return False


def AssessExpandCompoundAssign(node, source_code):

    if node.type == 'assignment_expression':
        compound_operators = {
            '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>=', '>>>='
        }
        has_compound_op = any(child.type in compound_operators for child in node.children)
        
        if has_compound_op:
            parent = node.parent
            if not parent:
                return False

            if parent.type == 'expression_statement':
                return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)

            if parent.type == 'for_statement' and parent.child_by_field_name('update') == node:
                return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)
            
    return False


def AssessExpandUnaryOP(node, source_code):

    if node.type == 'update_expression':
        parent = node.parent
        if not parent:
            return False

        if parent.type == 'expression_statement':
            return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)

        if parent.type == 'for_statement' and parent.child_by_field_name('update') == node:
            return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)
            
    return False


def AssessPromoteIntToLong(node, source_code):

    if node.type == 'local_variable_declaration':
        type_node = node.child_by_field_name('type')
        if type_node:
            type_text = source_code.encode('utf-8')[type_node.start_byte:type_node.end_byte]
            if type_text == b'int':
                for declarator in node.children_by_field_name('declarator'):
                    value_node = declarator.child_by_field_name('value')
                    if value_node and 'literal' in value_node.type:
                        return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)
    return False


def AssessPromoteFloatToDouble(node, source_code):

    if node.type == 'local_variable_declaration':
        type_node = node.child_by_field_name('type')
        if type_node:
            type_text = source_code.encode('utf-8')[type_node.start_byte:type_node.end_byte]
            if type_text == b'float':
                for declarator in node.children_by_field_name('declarator'):
                    value_node = declarator.child_by_field_name('value')
                    if value_node and 'literal' in value_node.type:
                        return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)
    return False


def AssessAddUnusedParameter(node, source_code):

    if node.type in ['method_declaration', 'constructor_declaration']:

        body_node = node.child_by_field_name('body')

        start_byte = node.start_byte
        start_point = node.start_point

        if body_node:

            end_byte = body_node.start_byte
            end_point = body_node.start_point
        else:
            end_byte = node.end_byte
            end_point = node.end_point

        return (True, start_byte, end_byte, start_point, end_point)

    return False


def AssessRefactorOutputAPI(node, source_code):

    if node.type != 'method_invocation':
        return False
    name_node = node.child_by_field_name('name')
    if not name_node:
        return False
    method_name = source_code[name_node.start_byte:name_node.end_byte]
    if method_name != 'println':
        return False
    object_node = node.child_by_field_name('object')
    if not object_node:
        return False
    object_name = source_code[object_node.start_byte:object_node.end_byte]
    if object_name == 'System.out':
        return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)
    return False


def AssessRenameVariable(node, source_code):

    if node.type != 'identifier':
        return False

    parent = node.parent
    if not parent:
        return False

    if parent.type == 'variable_declarator':
        grandparent = parent.parent

        if grandparent and grandparent.type in ['local_variable_declaration', 'field_declaration']:
            if parent.child_by_field_name('name') == node:
                return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)

    elif parent.type == 'formal_parameter':
        if parent.child_by_field_name('name') == node:
            return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)

    return False

def AssessRenameClassAndMethod(node, source_code):

    if node.type == 'class_declaration':
        name_node = node.child_by_field_name('name')
        if name_node:
            return (True, name_node.start_byte, name_node.end_byte, name_node.start_point, name_node.end_point)
    elif node.type == 'method_declaration':
        name_node = node.child_by_field_name('name')
        if name_node:
            return (True, name_node.start_byte, name_node.end_byte, name_node.start_point, name_node.end_point)
    return False