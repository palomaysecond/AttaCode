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
        return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)
    return False


def AssessWhileToForRefactoring_new(node, source_code):

    if node.type == 'while_statement':
        return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)
    return False


def AssessDoWhileToWhileConversion_new(node, source_code):

    if node.type == 'do_statement':
        return (True, node.start_byte, node.end_byte, node.start_point, node.end_point)
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
                        return (True, condition_node.start_byte, condition_node.end_byte, condition_node.start_point,
                                condition_node.end_point)
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
            return (
            True, prev_sibling.start_byte, parent_block.end_byte, prev_sibling.start_point, parent_block.end_point)
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


