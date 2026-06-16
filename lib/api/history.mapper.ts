// lib/api/history.mapper.ts

export interface UIClassificationHistory {
    id: string
    image: string
    createdDate: string
    subject: string
    confidence: number
    confidenceLabel: string
    gradeLabel: string
    modelVariant: string
  }
  
  const GRADE_LABEL_MAP: Record<number, string> = {
    10: "Lớp 10",
    11: "Lớp 11",
    12: "Lớp 12",
  }
  
  const DEFAULT_GRADE_LABEL = "Không xác định"
  
  export function mapHistoryItemToUI(item: {
    public_code: string
    image_path: string
    image_url: string
    subject: string
    confidence: number
    grade: number
    model_variant: string
    created_at: string
  }): UIClassificationHistory {
    return {
      id: item.public_code,
      image: item.image_url,
      subject: item.subject,
  
      confidence: item.confidence,
      confidenceLabel: `${(item.confidence * 100).toFixed(1)}%`,
  
      gradeLabel:
        GRADE_LABEL_MAP[item.grade] ?? DEFAULT_GRADE_LABEL,
  
      modelVariant: item.model_variant,
  
      createdDate: new Date(item.created_at).toLocaleString("vi-VN"),
    }
  }
  