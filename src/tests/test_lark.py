import os
import sys

from lark import Token
from lark.lexer import TerminalDef

import generative_approach

import lark

from generative_approach.lark_decoding import LarkDecoder
from generative_approach.lark_parser import LarkParser
from generative_approach.models.flan_t5 import FlanT5

test_doc = ('[start:Publication][start:describes][start:ClinicalTrial]['
            'start:analysesHealthCondition]▁pigmentary▁glaucoma[end:analysesHealthCondition][start:hasArm]['
            'start:Arm][start:hasAdverseEffect][start:Outcome][start:hasEndpoint][start:Endpoint]['
            'start:hasEndoPointDescription]▁conjunctival▁hyperemia[end:hasEndoPointDescription][end:Endpoint]['
            'end:hasEndpoint][start:hasResultMeasuredValue]▁0▁.▁3[end:hasResultMeasuredValue][end:Outcome]['
            'end:hasAdverseEffect][start:hasAdverseEffect][start:Outcome][start:hasEndpoint][start:Endpoint]['
            'start:hasEndoPointDescription]▁change▁in▁iris▁color[end:hasEndoPointDescription][end:Endpoint]['
            'end:hasEndpoint][start:hasNumberAffected]▁1[end:hasNumberAffected][end:Outcome][end:hasAdverseEffect]['
            'start:hasIntervention][start:Intervention][start:hasFrequency]▁once▁daily[end:hasFrequency]['
            'start:hasMedication][start:Medication][start:hasDeliveryMethod]▁topically[end:hasDeliveryMethod]['
            'start:hasDeliveryMethod]▁eyedrops[end:hasDeliveryMethod][start:hasDoseValue]▁0▁.▁005[end:hasDoseValue]['
            'start:hasDrug]▁latanoprost[end:hasDrug][end:Medication][end:hasMedication][start:hasMedication]['
            'start:Medication][start:hasDeliveryMethod]▁eyedrops[end:hasDeliveryMethod][start:hasDrug]▁placebo['
            'end:hasDrug][end:Medication][end:hasMedication][end:Intervention][end:hasIntervention]['
            'start:hasIntervention][start:Intervention][start:hasFrequency]▁once▁daily[end:hasFrequency]['
            'start:hasMedication][start:Medication][start:hasDeliveryMethod]▁eyedrops[end:hasDeliveryMethod]['
            'start:hasDrug]▁placebo[end:hasDrug][end:Medication][end:hasMedication][end:Intervention]['
            'end:hasIntervention][start:hasNumberPatientsArm]▁18[end:hasNumberPatientsArm][start:hasOutcome]['
            'start:Outcome][start:hasChangeValue]▁6▁.▁0[end:hasChangeValue][start:hasEndpoint][start:Endpoint]['
            'start:hasAggregationMethod]▁Mean[end:hasAggregationMethod][start:hasBaselineUnit]▁%['
            'end:hasBaselineUnit][start:hasEndoPointDescription]▁intraocular▁pressure[end:hasEndoPointDescription]['
            'end:Endpoint][end:hasEndpoint][start:hasSdErrorChangeValue]▁4▁.▁5[end:hasSdErrorChangeValue]['
            'start:hasTimePoint]▁6▁and▁12▁months[end:hasTimePoint][end:Outcome][end:hasOutcome][start:hasOutcome]['
            'start:Outcome][start:hasChangeValue]▁5▁.▁9[end:hasChangeValue][start:hasEndpoint][start:Endpoint]['
            'start:hasAggregationMethod]▁Mean[end:hasAggregationMethod][start:hasBaselineUnit]▁%['
            'end:hasBaselineUnit][start:hasEndoPointDescription]▁intraocular▁pressure[end:hasEndoPointDescription]['
            'end:Endpoint][end:hasEndpoint][start:hasSdErrorChangeValue]▁4▁.▁6[end:hasSdErrorChangeValue]['
            'start:hasTimePoint]▁12▁months[end:hasTimePoint][end:Outcome][end:hasOutcome][start:hasOutcome]['
            'start:Outcome][start:hasEndpoint][start:Endpoint][start:hasAggregationMethod]▁Mean['
            'end:hasAggregationMethod][start:hasBaselineUnit]▁%[end:hasBaselineUnit]['
            'start:hasEndoPointDescription]▁Outflow▁facility[end:hasEndoPointDescription]['
            'start:hasMeasurementDevice]▁Schiotz▁electronic▁tonometer[end:hasMeasurementDevice][end:Endpoint]['
            'end:hasEndpoint][start:hasPValueChangeValue]▁P▁=▁0▁.▁017[end:hasPValueChangeValue]['
            'start:hasRelativeChangeValue]▁30[end:hasRelativeChangeValue][end:Outcome][end:hasOutcome][end:Arm]['
            'end:hasArm][start:hasArm][start:Arm][start:hasAdverseEffect][start:Outcome][start:hasEndpoint]['
            'start:Endpoint][start:hasEndoPointDescription]▁conjunctival▁hyperemia[end:hasEndoPointDescription]['
            'end:Endpoint][end:hasEndpoint][start:hasResultMeasuredValue]▁0▁.▁2[end:hasResultMeasuredValue]['
            'end:Outcome][end:hasAdverseEffect][start:hasAdverseEffect][start:Outcome][start:hasEndpoint]['
            'start:Endpoint][start:hasEndoPointDescription]▁change▁in▁iris▁color[end:hasEndoPointDescription]['
            'end:Endpoint][end:hasEndpoint][start:hasNumberAffected]▁none[end:hasNumberAffected][end:Outcome]['
            'end:hasAdverseEffect][start:hasAdverseEffect][start:Outcome][start:hasBaselineValue]▁72['
            'end:hasBaselineValue][start:hasEndpoint][start:Endpoint][start:hasBaselineUnit]▁beats▁per▁minute['
            'end:hasBaselineUnit][start:hasEndoPointDescription]▁heart▁rate[end:hasEndoPointDescription]['
            'end:Endpoint][end:hasEndpoint][start:hasResultMeasuredValue]▁67[end:hasResultMeasuredValue]['
            'start:hasTimePoint]▁12▁months[end:hasTimePoint][end:Outcome][end:hasAdverseEffect]['
            'start:hasIntervention][start:Intervention][start:hasFrequency]▁twice▁daily[end:hasFrequency]['
            'start:hasMedication][start:Medication][start:hasDeliveryMethod]▁eyedrops[end:hasDeliveryMethod]['
            'start:hasDoseUnit]▁%[end:hasDoseUnit][start:hasDoseValue]▁0▁.▁5[end:hasDoseValue]['
            'start:hasDrug]▁timolol[end:hasDrug][end:Medication][end:hasMedication][end:Intervention]['
            'end:hasIntervention][start:hasNumberPatientsArm]▁18[end:hasNumberPatientsArm][start:hasOutcome]['
            'start:Outcome][start:hasChangeValue]▁4▁.▁8[end:hasChangeValue][start:hasEndpoint][start:Endpoint]['
            'start:hasAggregationMethod]▁Mean[end:hasAggregationMethod][start:hasBaselineUnit]▁%['
            'end:hasBaselineUnit][start:hasEndoPointDescription]▁intraocular▁pressure[end:hasEndoPointDescription]['
            'end:Endpoint][end:hasEndpoint][start:hasSdErrorChangeValue]▁3▁.▁0[end:hasSdErrorChangeValue]['
            'start:hasTimePoint]▁6▁and▁12▁months[end:hasTimePoint][end:Outcome][end:hasOutcome][start:hasOutcome]['
            'start:Outcome][start:hasChangeValue]▁4▁.▁6[end:hasChangeValue][start:hasEndpoint][start:Endpoint]['
            'start:hasAggregationMethod]▁Mean[end:hasAggregationMethod][start:hasBaselineUnit]▁%['
            'end:hasBaselineUnit][start:hasEndoPointDescription]▁intraocular▁pressure[end:hasEndoPointDescription]['
            'end:Endpoint][end:hasEndpoint][start:hasSdErrorChangeValue]▁3▁.▁1[end:hasSdErrorChangeValue]['
            'start:hasTimePoint]▁12▁months[end:hasTimePoint][end:Outcome][end:hasOutcome][end:Arm][end:hasArm]['
            'start:hasCTDesign]▁randomized[end:hasCTDesign][start:hasCTDesign]▁double▁-▁masked[end:hasCTDesign]['
            'start:hasCTDesign]▁Prospective[end:hasCTDesign][start:hasCTduration]▁12▁-▁month[end:hasCTduration]['
            'start:hasConclusionComment]▁Although▁further▁studies▁may▁need▁to▁confirm▁these▁data▁on▁a▁larger▁sample'
            '▁and▁to▁evaluate▁the▁side▁effect▁of▁increased▁iris▁pigmentation▁on▁long▁-▁term▁follow▁-▁up▁,'
            '▁in▁patients▁with▁pigmentary▁glaucoma▁,▁0▁.▁005▁%▁latanoprost▁taken▁once▁daily▁was['
            'end:hasConclusionComment][start:hasConclusionComment]▁well▁tolerated▁and▁more▁effective▁in▁reducing▁IOP'
            '▁than▁0▁.▁5▁%▁timolol▁taken▁twice▁daily▁.[end:hasConclusionComment][start:hasDiffBetweenGroups]['
            'start:DiffBetweenGroups][start:hasPvalueDiff]▁P▁<▁0▁.▁001[end:hasPvalueDiff][end:DiffBetweenGroups]['
            'end:hasDiffBetweenGroups][start:hasDiffBetweenGroups][start:DiffBetweenGroups]['
            'start:hasPvalueDiff]▁P▁<▁0▁.▁001[end:hasPvalueDiff][end:DiffBetweenGroups][end:hasDiffBetweenGroups]['
            'start:hasNumberPatientsCT]▁Thirty▁-▁six[end:hasNumberPatientsCT]['
            'start:hasObjectiveDescription]▁To▁compare▁the▁efficacy▁and▁side▁effects▁and▁the▁effect▁on▁aqueous▁humor'
            '▁dynamics▁of▁0▁.▁005▁%▁latanoprost▁applied▁topically▁once▁daily▁with▁0▁.▁5▁%▁timolol▁given▁twice▁daily'
            '▁for▁12▁months▁to▁patients▁with▁pigmentary▁glaucoma▁.[end:hasObjectiveDescription][end:ClinicalTrial]['
            'end:describes][start:hasAuthor]▁Mastropasqua▁L[end:hasAuthor][start:hasAuthor]▁Carpineto▁P['
            'end:hasAuthor][start:hasAuthor]▁Ciancaglini▁M[end:hasAuthor][start:hasAuthor]▁Gallenga▁PE['
            'end:hasAuthor][start:hasJournal]▁Ophthalmology[end:hasJournal][start:hasPMID]▁10080213[end:hasPMID]['
            'start:hasPublicationYear]▁1999[end:hasPublicationYear][start:hasPublicationYear]▁twice▁daily['
            'end:hasPublicationYear][start:hasTitle]▁A▁12▁-▁month▁,▁randomized▁,'
            '▁double▁-▁masked▁study▁comparing▁latanoprost▁with▁timolol▁in▁pigmentary▁glaucoma▁.[end:hasTitle]['
            'end:Publication]')

def test_lark():
    # https://stackoverflow.com/questions/68022036/getting-next-possible-tokens-with-lark-parsing

    #grammar = open(os.path.join(os.path.dirname(sys.modules["generative_approach"].__file__), "resources", 'grammar_full.lark')).read()
    #vocab_rule = "\nPOINT: /" + LarkDecoder.create_not_string_regex("[start:") + "|" + LarkDecoder.create_not_string_regex("[end:") + "/"
    #grammar += vocab_rule
    #parser = lark.Lark(grammar, start='publication', parser='lalr')
    model = FlanT5(model_name="google/flan-t5-small", device="cpu")
    lp = LarkParser(tokenizer=model.tokenizer)
    parser = lp.parser

    interactive = parser.parse_interactive("[start:Publication][start:describes][start:ClinicalTrial][start:analysesHealthCondition]▁type▁2▁diabetes[end:analysesHealthCondition][start:hasArm][start:Arm][start:hasAdverseEffect][start:Outcome][start:hasEndpoint][start:Endpoint][start:hasEndoPointDescription]▁Treatment▁-▁emergent▁adverse▁events[end:hasEndoPointDescription][start:hasEndoPointDescription]▁63▁.▁8[end:hasEndoPointDescription][start:hasEndoPointDescription]▁Treatment▁-▁emergent▁adverse▁events[end:hasEndoPointDescription][start:hasEndoPointDescription]▁40▁.▁8[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]▁symptomatic▁hypoglycaemia[end:hasEndoPointDescription][start:hasEndoPointDescription]")
    # feeds the text given to above into the parsers. This is not done automatically.
    interactive.exhaust_lexer()

    lres = list(parser.lex("▁[end:hasEndoPointDescription][end:Endpoint][end:hasEndpoint][end:Outcome][end:hasAdverseEffect][end:Arm][end:hasArm][end:ClinicalTrial][end:describes][end:Publication]"))

    while interactive.accepts() != {"$END"}:
        token_candidates = [x.pattern.value for x in parser.terminals if x.name in interactive.accepts() and x.pattern.value != "POINT" and x.pattern.value.startswith("[end:")]
        if len(token_candidates) > 0:
            for tok in parser.lex(token_candidates[0]):
                interactive.feed_token(tok)
        else:  # max length reached just when POINT schould have been generated, add dummy POINT
            for tok in parser.lex("<unk>"):
                interactive.feed_token(tok)
        interactive.exhaust_lexer()

    # returns the names of the Terminals that are currently accepted.
    print([x.pattern for x in parser.terminals if x.name in interactive.accepts()])

    for tok in parser.lex("▁pigmentary▁glaucoma"):
        interactive.feed_token(tok)

    # feeds the text given to above into the parsers. This is not done automatically.
    interactive.exhaust_lexer()

    # returns the names of the Terminals that are currently accepted.
    print([x.pattern for x in parser.terminals if x.name in interactive.accepts()])

    for tok in parser.lex("▁pigmentary▁glaucoma"):
        interactive.feed_token(tok)

    # feeds the text given to above into the parsers. This is not done automatically.
    interactive.exhaust_lexer()

    # returns the names of the Terminals that are currently accepted.
    print([x.pattern for x in parser.terminals if x.name in interactive.accepts()])

    tree = parser.parse(test_doc)
    print(tree.pretty())
    pass

def test_lark_parser():
    model = FlanT5(model_name="google/flan-t5-small", device="cpu")

    parser = LarkParser(
        grammar_file=os.path.join(os.path.dirname(sys.modules["generative_approach"].__file__), "resources", 'grammar_full.lark'),
        tokenizer=model.tokenizer
    )#.parse(test_doc)
    #print(parser.parser.parse(test_doc).pretty())
    res = parser.parse(test_doc)

    # print(test_doc)
    # res = parser.parser.lex(test_doc)
    #
    interactive = parser.parser.parse_interactive("[start:Publication][start:describes][start:ClinicalTrial][start:analysesHealthCondition]")
    # feeds the text given to above into the parsers. This is not done automatically.
    interactive.exhaust_lexer()
    print(interactive.accepts())
    pass
