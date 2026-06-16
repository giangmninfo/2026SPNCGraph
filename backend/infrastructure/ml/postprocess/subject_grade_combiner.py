from backend.infrastructure.ml.postprocess.subject_grade_combiner_product import combine_subject_grade_product
from backend.infrastructure.ml.postprocess.subject_grade_combiner_sqrt import combine_subject_grade_sqrt
from backend.infrastructure.ml.postprocess.subject_grade_combiner_weighted_geo import combine_subject_grade_weighted_geo

def combine_subject_grade(subjects, grades, method="product", **kwargs):
    if method == "product":
        return combine_subject_grade_product(subjects, grades, **kwargs)
    elif method == "sqrt":
        return combine_subject_grade_sqrt(subjects, grades, **kwargs)
    elif method == "weighted":
        return combine_subject_grade_weighted_geo(subjects, grades, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
