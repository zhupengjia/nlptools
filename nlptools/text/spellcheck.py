#!/usr/bin/env python


class SymSpellCorrection:
    """
        Use SymSpell for correction
    """
    def __init__(self, dictionary_path, term_index=0, count_index=1, max_edit_distance_dictionary=0, prefix_length=7, **args):
        """
        Input:
            - dictionary_path: string
            - term_index: int, column of the term in the dictionary text file, default is 0
            - count_index: int, column of the term frequency in the dictionary text file, default is 1
            - max_edit_distance_dictionary: int, maximum edit distance per dictionary precalculation, default is 0
            - prefix_length, int, default is 7
        """
        from symspellpy.symspellpy import SymSpell
        self.sym_spell = SymSpell(max_edit_distance_dictionary, prefix_length)
        self.sym_spell.load_dictionary(dictionary_path, term_index, count_index)

    def __call__(self, sentence):
        """
            Input:
                - sentence: string

            Output:
                - string
        """
        if len(sentence) < 1:
            return sentence
        try:
            corrected = self.sym_spell.word_segmentation(sentence).corrected_string
        except:
            print("Error spell correction:", sentence)
            corrected = sentence
        return corrected

class SpellCorrection:
    def __new__(cls, choose="symspellpy", **args):
        """
        TODO: now only support symspellpy
        """
        return SymSpellCorrection(**args)
