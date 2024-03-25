import json


class SantoToken:
    def __init__(self):
        self.sentence_number = None

        self.doc_char_onset = None

        self.doc_char_offset = None

        self.token = None

    def to_dict(self, simplified=False):
        return {
            "SantoToken": {
                "sentence_number": self.sentence_number,
                "doc_char_onset": self.doc_char_onset,
                "doc_char_offset": self.doc_char_offset,
                "token": self.token,
            }
        }

    def to_json(self, simplified=False):
        return json.dumps(self.to_dict(simplified=simplified), indent=2)

    @classmethod
    def from_dict(cls, d):
        assert len(d) == 1 and "SantoToken" in d
        d = d["SantoToken"]
        s = cls()
        s.sentence_number = d["sentence_number"]
        s.doc_char_onset = d["doc_char_onset"]
        s.doc_char_offset = d["doc_char_offset"]
        s.token = d["token"]
        return s

    @classmethod
    def from_json(cls, s):
        return cls.from_dict(json.loads(s))
