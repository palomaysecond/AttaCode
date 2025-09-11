import numpy as np

def get_descendants(node):
    descendants = []
    nodes_to_visit = [node]
    while nodes_to_visit:
        current = nodes_to_visit.pop()
        children = list(current.children)
        descendants.extend(children)
        nodes_to_visit.extend(children)
    return descendants


def AssessForToWhileConversion_new(node, source_code):

    if node.type in ['for_statement', 'for_range_loop']:
        return (True, node.start_byte , node.end_byte, node.start_point, node.end_point)
    return False

def AssessInlineLoopDeclaration_new(node, source_code):

    if node.type != 'declaration':
        return False
    declared_vars = set()
    for declarator_node in node.children_by_field_name('declarator'):
        if declarator_node.type == 'init_declarator':
            value_node = declarator_node.child_by_field_name('value')
            if value_node is None:
                name_node = declarator_node.child_by_field_name('declarator')
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
                    body_node = for_node.child_by_field_name('body')
                    end_byte = body_node.start_byte if body_node else for_node.end_byte
                    end_point = body_node.start_point if body_node else for_node.end_point
                    return (True, node.start_byte, end_byte, node.start_point, end_point)
    return False

def AssessExtractLoopDeclaration_new(node, source_code):

    if node.type == 'for_statement':
        init_node = node.child_by_field_name('init')
        if init_node and init_node.type == 'declaration':
            body_node = node.child_by_field_name('body')
            if body_node:
                r_paren_node = node.child_by_field_name('condition').next_sibling
                while r_paren_node and r_paren_node.type != ')':
                    r_paren_node = r_paren_node.next_sibling
                if r_paren_node and r_paren_node.type == ')':
                    end_byte = r_paren_node.end_byte
                    return (True, node.start_byte, end_byte, node.start_point, (body_node.start_point[0], body_node.start_point[1]))
    return False

def AssessIfElseBranchSwap_new(node, source_code):

    if node.type == 'if_statement':
        consequence_node = node.child_by_field_name('consequence')
        alternative_node = node.child_by_field_name('alternative')
        if not (consequence_node and alternative_node):
            return False
        condition_node = node.child_by_field_name('condition')
        if condition_node and condition_node.type == 'parenthesized_expression':
            actual_condition = condition_node.named_children[0]
            if actual_condition.type == 'binary_expression':
                for child in actual_condition.children:
                    if child.type in ['==', '!=']:
                        return (True, condition_node.start_byte, condition_node.end_byte, condition_node.start_point, condition_node.end_point)
    return False

def AssessElseIfToNestedIf_new(node, source_code):

    if node.type != 'if_statement':
        return False
    consequence_node = node.child_by_field_name('consequence')
    if not (consequence_node and consequence_node.type == 'compound_statement'):
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

def AssessNestedIfToElseIf_new(node, source_code):

    if node.type != 'if_statement':
        return False
    consequence_node = node.child_by_field_name('consequence')
    if not (consequence_node and consequence_node.type == 'compound_statement'):
        return False
    parent_block = node.parent
    if not parent_block or parent_block.type != 'compound_statement' or parent_block.named_child_count != 1:
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

def AssessUnwrapRedundantBlock_new(node, source_code):

    if node.type == 'if_statement':
        consequence = node.child_by_field_name('consequence')
        if consequence and consequence.type == 'compound_statement':
            return (True, consequence.start_byte, consequence.end_byte, consequence.start_point, consequence.end_point)
    return False

def AssessWrapStatementInBlock_new(node, source_code):

    if node.type == 'if_statement':
        consequence = node.child_by_field_name('consequence')
        if consequence and consequence.type != 'compound_statement':
            return (True, consequence.start_byte, consequence.end_byte, consequence.start_point, consequence.end_point)
    return False

def AssessSplitCompoundCondition_new(node, source_code):

    if node.type == 'if_statement':
        consequence_node = node.child_by_field_name('consequence')
        if not (consequence_node and consequence_node.type == 'compound_statement'):
            return False
        condition_node = node.child_by_field_name('condition')
        if condition_node and condition_node.type == 'parenthesized_expression':
            actual_condition = condition_node.named_children[0]
            if actual_condition.type == 'binary_expression':
                for child in actual_condition.children:
                    if child.type in ['&&', '||']:
                        return (True, condition_node.start_byte, condition_node.end_byte, condition_node.start_point, condition_node.end_point)
    return False

def AssessAddRedundantStatement_new(node, source_code):

    if node.type == 'declaration':
        return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)
    return False

def AssessWrapWithConstantCondition_new(node, source_code):

    if node.type != 'expression_statement':
        return False
    if not (node.named_child_count > 0 and node.named_children[0].type == 'assignment_expression'):
        return False
    if node.parent and node.parent.type == 'compound_statement':
        return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)
    return False

def AssessExtractSubexpression_new(node, source_code):

    if not (node.type == 'expression_statement' and node.parent and node.parent.type == 'compound_statement'):
        return False
    if not (node.named_child_count > 0 and node.named_children[0].type == 'assignment_expression'):
        return False
    assignment_node = node.named_children[0]
    rhs_node = assignment_node.child_by_field_name('right')
    if not (rhs_node and rhs_node.type == 'binary_expression'):
        return False
    left_operand = rhs_node.child_by_field_name('left')
    right_operand = rhs_node.child_by_field_name('right')
    if (left_operand and left_operand.type in ['binary_expression', 'call_expression']) or \
       (right_operand and right_operand.type in ['binary_expression', 'call_expression']):
        return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)
    return False

def AssessSwitchToIfElse_new(node, source_code):

    if node.type == 'switch_statement':
        return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)
    return False

def AssessReturnViaTempVariable_new(node, source_code):

    if node.type == 'return_statement':
        if node.named_child_count > 0:
            return_value_node = node.named_children[0]
            literal_types = {'number_literal', 'string_literal', 'char_literal', 'true', 'false', 'null'}
            if return_value_node.type in literal_types:
                return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)
    return False

def AssessPromoteIntToLong_new(node, source_code):

    if node.type == 'declaration':
        type_node = node.child_by_field_name('type')
        if type_node and type_node.type == 'primitive_type':
            type_text = source_code[type_node.start_byte:type_node.end_byte]
            if type_text == 'int':
                for declarator in node.children_by_field_name('declarator'):
                    if declarator.type == 'init_declarator':
                        value_node = declarator.child_by_field_name('value')
                        if value_node and value_node.type == 'number_literal':
                            return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)
    return False

def AssessPromoteFloatToDouble_new(node, source_code):

    if node.type == 'declaration':
        type_node = node.child_by_field_name('type')
        if type_node and type_node.type == 'primitive_type':
            type_text = source_code[type_node.start_byte:type_node.end_byte]
            if type_text == 'float':
                for declarator in node.children_by_field_name('declarator'):
                     if declarator.type == 'init_declarator':
                        value_node = declarator.child_by_field_name('value')
                        if value_node and value_node.type == 'number_literal':
                            return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)
    return False

def AssessAddUnusedParameter_new(node, source_code):

    if node.type == 'function_definition':
        body_node = node.child_by_field_name('body')
        if not body_node:
            for child in reversed(node.children):
                if child.type == 'compound_statement':
                    body_node = child
                    break

        if body_node:

            return (True, node.start_byte, body_node.start_byte, node.start_point, body_node.start_point)
        else:
            return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)

    if node.type in ['declaration', 'field_declaration']:

        q = list(node.children)
        while q:
            child = q.pop(0)
            if child.type == 'function_declarator':

                is_function_pointer = any(
                    d.type == 'parenthesized_declarator' for d in get_descendants(child)
                )
                if not is_function_pointer:
                    return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)

            q.extend(child.children)

    return False


def AssessRefactorOutputAPI_new(node, source_code):

    if node.type == 'call_expression':
        func_node = node.child_by_field_name('function')
        if func_node:
            # C: printf
            if func_node.type == 'identifier':
                func_name = source_code[func_node.start_byte:func_node.end_byte]
                if func_name == 'printf':
                    return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)
            # C++: std::cout << (represented as a call_expression with `<<` operator)
            elif func_node.type == 'field_expression':
                 if 'cout' in source_code[func_node.start_byte:func_node.end_byte]:
                    return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)
    return False

def AssessRenameVariable_new(node, source_code):

    if node.type != 'identifier':
        return False
    parent = node.parent
    if not parent:
        return False

    if parent.type == 'init_declarator' and parent.child_by_field_name('declarator') == node:
        grandparent = parent.parent
        if grandparent and grandparent.type == 'declaration':
            return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)

    elif parent.type == 'parameter_declaration' and parent.child_by_field_name('declarator') == node:
        return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)
    return False

def AssessRenameClassAndMethod_new(node, source_code):

    if node.type in ['class_specifier', 'struct_specifier']:
        name_node = node.child_by_field_name('name')
        if name_node:
            return (True, name_node.start_byte, name_node.end_byte, name_node.start_point, name_node.end_point)
    elif node.type == 'function_definition':
        declarator_node = node.child_by_field_name('declarator')
        if declarator_node:
            while declarator_node.child_by_field_name('declarator'):
                declarator_node = declarator_node.child_by_field_name('declarator')
            name_node = declarator_node.child_by_field_name('declarator')
            if name_node and name_node.type == 'identifier':
                 return (True, name_node.start_byte, name_node.end_byte, name_node.start_point, name_node.end_point)
    return False
def AssessWhileToForRefactoring_new(node, source_code):

    if node.type == 'while_statement':
        return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)
    return False


def AssessDoWhileToWhileConversion_new(node, source_code):

    if node.type == 'do_statement':
        return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)
    return False


def AssessNegateWithReversedOperator_new(node, source_code):

    if node.type == 'binary_expression':
        comparison_operators = {'<', '>', '<=', '>=', '==', '!='}
        for child in node.children:
            if child.type in comparison_operators:
                return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)
    return False


def AssessExpandCompoundAssign_new(node, source_code):

    if node.type == 'assignment_expression':
        compound_operators = {
            '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>='
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


def AssessExpandUnaryOP_new(node, source_code):

    if node.type == 'update_expression':
        parent = node.parent
        if not parent:
            return False

        if parent.type == 'expression_statement':
            return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)

        if parent.type == 'for_statement' and parent.child_by_field_name('update') == node:
            return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)

    return False
