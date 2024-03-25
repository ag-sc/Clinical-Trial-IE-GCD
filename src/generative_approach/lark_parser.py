import os
import sys
from collections import defaultdict
from pprint import pprint
from queue import Queue
from typing import Optional

import lark
from transformers import PreTrainedTokenizer

from template_lib.data_classes.Template import Template
from template_lib.data_classes.TemplateCollection import TemplateCollection


class LarkParser:
    names = ["Arm", "ClinicalTrial", "DiffBetweenGroups", "Endpoint", "Intervention", "Medication", "Outcome",
             "Population", "Publication", "analysesHealthCondition", "describes", "hasAdverseEffect",
             "hasAggregationMethod", "hasAllocationRatio", "hasArm", "hasAuthor", "hasAvgAge", "hasBaselineUnit",
             "hasBaselineValue", "hasCTDesign", "hasCTduration", "hasChangeValue", "hasConclusionComment",
             "hasConfIntervalChangeValue", "hasConfIntervalDiff", "hasCountry", "hasDeliveryMethod",
             "hasDiffBetweenGroups", "hasDiffGroupAbsValue", "hasDoseDescription", "hasDoseUnit", "hasDoseValue",
             "hasDrug", "hasEndoPointDescription", "hasEndpoint", "hasFinalNumPatientsArm", "hasFinalNumberPatientsCT",
             "hasFrequency", "hasIntervention", "hasJournal", "hasMeasurementDevice", "hasMedication", "hasMinAge",
             "hasNumberAffected", "hasNumberPatientsArm", "hasNumberPatientsCT", "hasObjectiveDescription",
             "hasObservedResult", "hasOutcome", "hasPMID", "hasPValueChangeValue", "hasPercentageAffected",
             "hasPopulation", "hasPrecondition", "hasPublicationYear", "hasPvalueDiff", "hasRelativeChangeValue",
             "hasRelativeFreqTime", "hasResultMeasuredValue", "hasSdDevBL", "hasSdDevChangeValue", "hasSdDevResValue",
             "hasSdErrorChangeValue", "hasSubGroupDescription", "hasTimePoint", "hasTitle"]

    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 grammar_file: Optional[str] = None,
                 ):
        if grammar_file is None:
            grammar_file = os.path.join(os.path.dirname(sys.modules["generative_approach"].__file__), "resources", 'grammar_full.lark')
        self.tokenizer = tokenizer
        self.grammar = open(grammar_file).read()

        len_sorted_vocab = defaultdict(set)
        danger_vocab = set()

        for v in self.tokenizer.get_vocab().keys():
            if "[" in v:
                danger_vocab.add(v)
            else:
                len_sorted_vocab[len(v)].add(v)

        len_sorted_vocab[1].add("<")
        len_sorted_vocab[1].add("[")
        vocab_rule = ""

        lens = sorted(len_sorted_vocab.keys(), reverse=True)
        maxlen = lens[0]

        for l in lens:
            vocab = len_sorted_vocab[l]
            vocab_rule += f"\nPOINT{l}.-{maxlen-l+1}: " + self.vocab_regex(vocab)

        #vocab_rule += f"\nPOINT0.-{maxlen+1}: " + self.vocab_regex(danger_vocab)

        vocab_rule += "\nPOINT.-1: " + "|".join([f"POINT{l}" for l in lens])# + "|POINT0"

        #print(vocab_rule)

        #'"' + k.replace('"', '["]') + '"'

        # vocab_rule = "\nPOINT.-1: " + " | ".join(["".join([
        #     '"' + k.replace('"', '\\"') + '"' for c in k
        # ]) for k in vocab])

        self.grammar += vocab_rule
        #print(self.grammar)
        self.parser = lark.Lark(self.grammar, start='publication', parser="lalr", lexer="contextual")
        self._camelize = {name.lower(): name for name in self.names}

    @staticmethod
    def vocab_regex(vocab):
        return "/" + "|".join(["".join([
            "[" + c.replace("\\", "\\\\").replace("/", "\\/").replace("]", "\\]") + "]" #if c != "]" else c
            for c in k  # if c != "\n" else "\\n"
        ]) for k in vocab]) + "/"

    def camelize(self, name: str):
        if name.lower() in self._camelize.keys():
            return self._camelize[name.lower()]
        else:
            return name

    def _tree_to_dict(self, tree: lark.Tree):
        d = defaultdict(list)
        for child in tree.children:
            if isinstance(child, lark.Token):
                d[self.camelize(child.type)].append(child.value)
            else:
                d[self.camelize(child.data.value)].append(self._tree_to_dict(child))

        if "POINT" in d.keys():
            return d["POINT"]#"".join(d["POINT"])
            #return {"Entity": "".join(d["POINT"])}
        else:
            return dict(d)

    def parse(self, text: str) -> TemplateCollection:#Template


        tree = self.parser.parse(text)
        dict_res = {"Publication": [self._tree_to_dict(tree)]}
        #print()
        print(dict_res)

        pub = Template.from_dict(dict_res, simplified=True)
        tc = TemplateCollection()
        tc.add_template(pub)
        q = Queue()
        q.put(pub)
        seen = set()
        seen.add(pub.get_id())
        while not q.empty():
            t = q.get()
            for sn, sfs in t.get_slots().items():
                for sf in sfs:
                    if isinstance(sf, Template) and sf.get_id() not in seen:
                        q.put(sf)
                        seen.add(sf.get_id())
                        tc.add_template(sf)

        return tc


