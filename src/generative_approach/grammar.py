import sys

from generative_approach.utils import *


def create_nonterminal_name(name, head: bool = False):
    if head:
        return '#' + name.upper() + '_HEAD'
    else:
        return '#' + name.upper()


class GrammarRule:
    def __init__(self, rule_definition: str, position: int):
        [rule_name_cardinality, symbols] = rule_definition.split('::')
        rule_name_cardinality = rule_name_cardinality.strip()
        symbols = symbols.strip()

        self._symbols = symbols.split(' ')
        self._position = position

        if rule_name_cardinality.count('{') > 0:
            self._rule_name, cardinality = rule_name_cardinality.split('_')
            cardinality = cardinality.replace('{', '')
            cardinality = cardinality.replace('}', '')

            min_cardinality, max_cardinality = cardinality.split(',')
            self._min_cardinality = int(min_cardinality)
            self._max_cardinality = int(max_cardinality)
        else:
            self._rule_name = rule_name_cardinality
            self._min_cardinality = None
            self._max_cardinality = None

    def get_symbols(self):
        return self._symbols.copy()

    def __getitem__(self, index):
        return self._symbols[index]

    def __setitem__(self, index, value):
        self._symbols[index] = value

    def __iter__(self):
        return iter(self._symbols)

    def get_rule_name(self):
        return self._rule_name

    def get_position(self):
        return self._position

    def get_min_cardinality(self):
        return self._min_cardinality

    def get_max_cardinality(self):
        return self._max_cardinality

    def print_ount(self):
        print('rule name: ', self.get_rule_name())
        print('rule symbols: ', self.get_symbols())
        print('position: ', self.get_position())
        print('min cardinality: ', self.get_min_cardinality())
        print('max cardinality: ', self.get_max_cardinality())


class Grammar:
    def __init__(
            self,
            filename: str,
            pointer_symbol: str = 'POINT'
    ):
        with open(filename) as fp:
            lines = map(lambda x: x.strip(), fp.readlines())

        # discard empty lines
        lines = filter(lambda x: len(x) > 0, lines)

        # parse rules of grammar
        grammar_rules = [GrammarRule(line, pos) for pos, line in enumerate(lines)]

        # group rules by rule name
        grammar_rules_dict = dict()
        for rule in grammar_rules:
            grammar_rules_dict.setdefault(rule.get_rule_name(), []).append(rule)

        self.pointer_symbol = pointer_symbol
        self._grammar_rules_dict = grammar_rules_dict

    def get_rule_start_symbols(self, rule_name):
        return [rule[0] for rule in self._grammar_rules_dict[rule_name]]

    def get_nonterminal_symbols(self) -> Set[str]:
        return set(self._grammar_rules_dict.keys())

    def __getitem__(self, rule_name):
        if rule_name not in self._grammar_rules_dict:
            raise KeyError('Unknown grammar rule: ' + str(rule_name))

        return self._grammar_rules_dict[rule_name]

    def get_rule_option_by_start_symbol(self, rule_name, start_symbol):
        for rule in self[rule_name]:
            if rule._symbols[0] == start_symbol:
                return rule

        raise Exception(f'no matching rule found for rule name {rule_name} and start symbol {start_symbol}')

    def convert_terminal_tokens_to_ids(self, tokenizer):
        control_symbols = self.get_nonterminal_symbols()
        control_symbols.add(self.pointer_symbol)

        for grammar_rule in chain(*list(self._grammar_rules_dict.values())):
            for i, symbol in enumerate(grammar_rule):
                if symbol not in control_symbols:
                    token_id = tokenizer.convert_tokens_to_ids([symbol])[0]

                    # replace terminal token by id
                    grammar_rule[i] = token_id

    def print_out(self):
        for rule_name in self._grammar_rules_dict:
            print('rule group name: ', rule_name)

            for rule in self._grammar_rules_dict[rule_name]:
                rule.print_ount()

            print('-------------')


def create_templates_grammar_file(
        filename: str,
        task_config_dict,
        flat: bool = False
):
    with open(filename, 'w') as fp:
        for template_name in sorted(task_config_dict['slots_ordering_dict'].keys()):
            rule_name = create_nonterminal_name(template_name)
            head_rule_name = create_nonterminal_name(template_name, True)

            # add head rule
            fp.write(
                head_rule_name
                + ' :: '
                + create_start_token(template_name)
                + ' ' + rule_name
                + '\n'
            )

            # add rules for slots of template
            for slot_name in task_config_dict['slots_ordering_dict'][template_name]:
                if slot_name not in task_config_dict['used_slots']:
                    continue

                if slot_name in task_config_dict['slots_containing_templates']:
                    if not flat:
                        fp.write(
                            rule_name
                            + ' :: '
                            + create_start_token(slot_name)
                            + ' ' + create_nonterminal_name(task_config_dict['slots_containing_templates'][slot_name],
                                                            head=True)
                            + ' '
                            + create_end_token(slot_name)
                            + ' ' + rule_name
                            + '\n'
                        )

                    continue

                # textual slot
                fp.write(
                    rule_name
                    + ' :: '
                    + create_start_token(slot_name)
                    + ' POINT '
                    + create_end_token(slot_name)
                    + ' ' + rule_name
                    + '\n'
                )

            # add end rule for template
            fp.write(
                rule_name
                + ' :: '
                + create_end_token(template_name)
                + '\n'
            )
            fp.write('\n')


if __name__ == '__main__':
    task_config_dict = import_task_config(sys.argv[1])
    create_templates_grammar_file('aaa.txt', task_config_dict=task_config_dict, flat=False)
