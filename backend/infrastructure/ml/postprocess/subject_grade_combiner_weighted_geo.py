def combine_subject_grade_weighted_geo(
    subjects,
    grades,
    topk=5,
    alpha=0.7,
    beta=0.3
):
    """
    Weighted geometric mean.
    Allows unequal contribution based on model reliability.
    """
    pairs = [
        (f"{s} - {g}", (sp ** alpha) * (gp ** beta))
        for s, sp in subjects
        for g, gp in grades
    ]
    return sorted(pairs, key=lambda x: x[1], reverse=True)[:topk]
