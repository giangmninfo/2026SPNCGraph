# infrastructure/ml/postprocess/subject_grade_combiner.py

def combine_subject_grade_product(subjects, grades, topk=5):
    """
    Baseline method: direct probability product.
    Assumes equal reliability and calibration.
    """
    pairs = [
        (f"{s} - {g}", sp * gp)
        for s, sp in subjects
        for g, gp in grades
    ]
    return sorted(pairs, key=lambda x: x[1], reverse=True)[:topk]
