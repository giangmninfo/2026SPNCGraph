import type { ModelClassificationResponse } from "./model"
import type { ClassificationResult } from "@/lib/types/classification"

const TOP_K = 3
/**
 * Default fallback for unknown subjects
 */
const DEFAULT_CATEGORY = {
  name: "Khác",
  color: "#9CA3AF", // neutral gray
}

/**
 * Map by SUBJECT CODE (language-agnostic, stable)
 */
const GRADE_LABEL_MAP: Record<number, string> = {
  10: "Lớp 10",
  11: "Lớp 11",
  12: "Lớp 12",
}

const DEFAULT_GRADE_LABEL = "Không xác định"
export const SUBJECT_COLOR_MAP: Record<string, string> = {
  UNKNOWN: "#9CA3AF",          // Neutral gray

  // 🧪 Natural Sciences
  BIOLOGY: "#45B7D1",          // Blue-cyan
  CHEMISTRY: "#A78BFA",        // Purple
  PHYSICS: "#4ECDC4",          // Teal

  // 📐 Formal Sciences
  MATHEMATICS: "#6366F1",      // Indigo
  INFORMATICS: "#0EA5E9",      // Sky blue

  // 🌍 Social Sciences & Humanities
  HISTORY: "#F4A261",          // Warm orange
  GEOGRAPHY: "#FFA07A",        // Light salmon
  LITERATURE: "#EC4899",       // Rose

  // 🌐 Languages
  ENGLISH: "#10B981",          // Emerald

  // 🎨 Arts
  ART: "#F472B6",              // Pink
  MUSIC: "#FB7185",            // Soft red

  // 🧠 Life / Career / Civic
  TECHNOLOGY: "#64748B",       // Slate
  DEFENSE_EDU: "#6B7280",      // Dark gray
  CAREER_GUIDANCE: "#22C55E",  // Green
}


const DEFAULT_SUBJECT_COLOR = "#9CA3AF"

/**
 * Mapper from backend → UI model
 */
export function mapModelToClassificationResult(
  data: ModelClassificationResponse
): ClassificationResult {
  const category =
    GRADE_LABEL_MAP[data.grade] ?? DEFAULT_GRADE_LABEL

  const categoryColor =
    SUBJECT_COLOR_MAP[data.subject_code] ??
    DEFAULT_SUBJECT_COLOR

  return {
    subject: data.subject,
    confidence: data.confidence,

    // 🔥 grade = category, subject = color
    category,
    categoryColor,

    classId: data.grade,
    processingTime: data.processing_time_ms ?? 0,
    modelVersion: data.model_variant,

    topPredictions: data.top_predictions.slice(0, TOP_K).map((p) => ({
      subject: p.subject,
      classId: p.grade,

      category:
        GRADE_LABEL_MAP[p.grade] ?? DEFAULT_GRADE_LABEL,

      confidence: p.confidence,

      categoryColor:
        SUBJECT_COLOR_MAP[p.subject_code] ??
        DEFAULT_SUBJECT_COLOR,
    })),

    analysisMetrics: {
      accuracy: data.confidence,
      precision: data.confidence,
      recall: data.confidence,
      inferenceSpeed: 0,
    },

    technicalDetails: {
      imageSize: data.image?.image_size ?? "N/A",
      format: data.image?.image_format ?? "N/A",
      dimensions: data.dimension,
      graphNodes: data.graph_nodes,
      graphEdges: data.graph_edges,
    }
  }
}