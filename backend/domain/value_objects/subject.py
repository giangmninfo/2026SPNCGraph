from enum import Enum

class SubjectCode(str, Enum):
    UNKNOWN = "UNKNOWN"
    
    TECHNOLOGY = "TECHNOLOGY"
    DEFENSE_EDU = "DEFENSE_EDU"
    CAREER_GUIDANCE = "CAREER_GUIDANCE"
    CHEMISTRY = "CHEMISTRY"
    HISTORY = "HISTORY"
    ART = "ART"
    LITERATURE = "LITERATURE"
    BIOLOGY = "BIOLOGY"
    INFORMATICS = "INFORMATICS"
    ENGLISH = "ENGLISH"
    MATHEMATICS = "MATHEMATICS"
    PHYSICS = "PHYSICS"
    MUSIC = "MUSIC"
    GEOGRAPHY = "GEOGRAPHY"

SUBJECT_VI_TO_CODE = {
    "Công nghệ": SubjectCode.TECHNOLOGY,
    "Giáo dục quốc phòng và an ninh": SubjectCode.DEFENSE_EDU,
    "Hoạt động trải nghiệm, hướng nghiệp": SubjectCode.CAREER_GUIDANCE,
    "Hóa học": SubjectCode.CHEMISTRY,
    "Lịch sử": SubjectCode.HISTORY,
    "Mĩ thuật": SubjectCode.ART,
    "Ngữ Văn": SubjectCode.LITERATURE,
    "Sinh học": SubjectCode.BIOLOGY,
    "Tin học": SubjectCode.INFORMATICS,
    "Tiếng Anh": SubjectCode.ENGLISH,
    "Toán": SubjectCode.MATHEMATICS,
    "Vật Lý": SubjectCode.PHYSICS,
    "Âm nhạc": SubjectCode.MUSIC,
    "Địa lí": SubjectCode.GEOGRAPHY,
}