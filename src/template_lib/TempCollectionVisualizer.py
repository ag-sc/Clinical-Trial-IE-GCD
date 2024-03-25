class TempCollectionVisualizer:
    def __init__(self, slot_filler_separator=' | '):
        self._slot_filler_separator = slot_filler_separator

    def to_tex(self, template_collection):
        result_str = ''

        for template_type in sorted(template_collection.get_template_types()):
            for template in template_collection.get_templates_by_type(template_type):
                # begin tabular tex environment
                result_str += '\\begin{table} \n'
                result_str += '\\begin{tabular}{ l|l } \n'

                # table header
                result_str += '\\multicolumn{2}{c}{' + template.get_id() + '} \\\\ \n'
                result_str += '\\hline \n'

                # each slot_name,slot_fillers pair is row od table
                for slot_name in template.get_filled_slot_names():
                    # get slot fillers as strings
                    entity_strings = template.get_slot_fillers_as_strings(slot_name)

                    # create tex row
                    result_str += '{slot_name} & {slot_fillers} \\\\ \n'.format(slot_name=slot_name,
                                                                                slot_fillers=self._slot_filler_separator.join(
                                                                                    entity_strings))

                # end tabbular environment
                result_str += '\\end{tabular} \n'
                result_str += '\\end{table} \n'

        return result_str.replace('_', '\\_').replace('%', '\\%')
