export interface ClassificationResult {
    subject: string
    confidence: number
    category: string
    categoryColor: string
    classId: number
    processingTime: number
    modelVersion: string
  
    topPredictions: Array<{
      subject: string
      classId: number
      category: string
      confidence: number
      categoryColor: string
    }>
  
    analysisMetrics: {
      accuracy: number
      precision: number
      recall: number
      inferenceSpeed: number
    }
  
    technicalDetails: {
      imageSize: string
      dimensions: number
      format: string
      graphNodes: number
      graphEdges: number | null
    }
  }
  