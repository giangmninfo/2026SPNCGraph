from math import sqrt

def combine_subject_grade_sqrt(subjects, grades, topk=5):
    """
    Monotonic transformation of the product.
    Preserves ranking; changes only score scale.
    """
    pairs = [
        (f"{s} - {g}", sqrt(sp * gp))
        for s, sp in subjects
        for g, gp in grades
    ]
    return sorted(pairs, key=lambda x: x[1], reverse=True)[:topk]
