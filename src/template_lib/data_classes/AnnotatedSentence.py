class AnnotatedSentence:
    def __init__(self):
        # list of bert token ids representing sentence
        self.tokens = None

        # sentence id of abstract
        self.sentence_id = None

        # list of pairs of indices into self.tokens list representing
        # entity boundaries, both including
        self.entity_boundaries = []

        # list of annotations for each entity
        self.annotations = []


def extract_annotated_sentences_from_abstract(abstract, tokenizer):
    sentence_numbers = sorted(abstract.get_sentence_numbers())
    annotations = abstract.annotated_abstract.annotations

    annotated_sentences = []
    # [cls_id,sep_id] = tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])

    for sentence_number in sentence_numbers:
        tokens = abstract.get_sentence_tokens(sentence_number)
        sentence_text = ' '.join(tokens)

        tokens = tokenizer.tokenize(sentence_text)

        annotated_sentence = AnnotatedSentence()
        annotated_sentence.sentence_id = sentence_number
        annotated_sentence.tokens = tokens

        for annotation in annotations:
            if annotation.sentence_number != sentence_number:
                continue

            left_context_tokens = tokenizer.tokenize(' '.join(annotation.left_context))
            entity_tokens = tokenizer.tokenize(' '.join(annotation.tokens))
            right_context_tokens = tokenizer.tokenize(' '.join(annotation.right_context))

            assert tokens == left_context_tokens + entity_tokens + right_context_tokens

            # add annotation to annotated_sentence
            left_offset = len(left_context_tokens)
            right_offset = left_offset + len(entity_tokens) - 1

            annotated_sentence.entity_boundaries.append((left_offset, right_offset))
            annotated_sentence.annotations.append(annotation)

        annotated_sentences.append(annotated_sentence)

    return annotated_sentences
