// lib/api/model.ts
export type ModelVariant = "GraphSAGE-I_v2" | "kNN-Voting" | "GraphSAGE-E_kNN"

export interface ModelClassificationResponse {
  subject: string
  subject_code: string
  confidence: number
  grade: number
  label: string
  processing_time_ms?: number
  model_variant: string
  graph_nodes: number
  graph_edges: number | null
  dimension: number

  image?: {
    image_size: string
    image_format: string
    image_width?: number
    image_height?: number
  }

  top_predictions: Array<{
    subject: string
    subject_code: string
    confidence: number
    grade: number
    label: string
  }>
}

export async function classifyImage(payload: {
  file: File
  model: ModelVariant
}) {
  const token = localStorage.getItem("access_token")

  if (!token) {
    throw new Error("Not authenticated")
  }

  const formData = new FormData()
  formData.append("image", payload.file)

  const res = await fetch(
    `${process.env.NEXT_PUBLIC_API_URL}/model/classification`,
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`, // ✅ REQUIRED
        "X-Model-Variant": payload.model,
      },
      body: formData,
    }
  )

  const data = await res.json()

  if (!res.ok) {
    if (res.status === 401) {
      throw new Error("Session expired. Please log in again.")
    }
    throw new Error(data.error || "Model classification failed")
  }

  return data as ModelClassificationResponse
}

export interface ClassificationHistoryItem {
  id: number
  public_code: string
  image_path: string
  image_url: string
  label: string
  confidence: number
  subject: string
  subject_code: string
  grade: number
  model_variant: string
  created_at: string
}

export interface ClassificationHistoryResponse {
  items: ClassificationHistoryItem[]
  page: number
  limit: number
  total: number
  total_pages: number
  q?: string
}

export async function fetchClassificationHistory(params?: {
  page?: number
  q?: string
}): Promise<ClassificationHistoryResponse> {
  const token = localStorage.getItem("access_token")

  if (!token) {
    throw new Error("Not authenticated")
  }

  const searchParams = new URLSearchParams()

  if (params?.page) {
    searchParams.set("page", params.page.toString())
  }

  if (params?.q) {
    searchParams.set("q", params.q)
  }

  const res = await fetch(
    `${process.env.NEXT_PUBLIC_API_URL}/model/classifications?${searchParams.toString()}`,
    {
      method: "GET",
      headers: {
        Authorization: `Bearer ${token}`,
      },
    }
  )

  const data = await res.json()

  if (!res.ok) {
    if (res.status === 401) {
      throw new Error("Session expired. Please log in again.")
    }
    throw new Error(data.error || "Failed to fetch classification history")
  }

  return data as ClassificationHistoryResponse
}
