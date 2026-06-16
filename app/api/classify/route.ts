import { type NextRequest, NextResponse } from "next/server"

// Class definitions with colors
const CLASSES = [
  { id: 1, name: "One Class", color: "#FF6B6B" },
  { id: 2, name: "Two Class", color: "#4ECDC4" },
  { id: 3, name: "Three Class", color: "#45B7D1" },
  { id: 4, name: "Four Class", color: "#FFA07A" },
  { id: 5, name: "Five Class", color: "#98D8C8" },
  { id: 6, name: "Six Class", color: "#F7DC6F" },
  { id: 7, name: "Seven Class", color: "#BB8FCE" },
  { id: 8, name: "Eight Class", color: "#85C1E2" },
  { id: 9, name: "Nine Class", color: "#F8B88B" },
]

// Mock GNN classification - simulates a real model
const subjects = [
  {
    subject: "Golden Retriever",
    confidence: 0.94,
    classId: 1,
    category: "One Class",
    categoryColor: "#FF6B6B",
    processingTime: 1240,
    modelVersion: "GNN-v2.1.4",
    topPredictions: [
      { subject: "Golden Retriever", classId: 1, category: "One Class", confidence: 0.94, categoryColor: "#FF6B6B" },
      { subject: "Labrador Retriever", classId: 2, category: "Two Class", confidence: 0.87, categoryColor: "#4ECDC4" },
      { subject: "German Shepherd", classId: 3, category: "Three Class", confidence: 0.76, categoryColor: "#45B7D1" },
    ],
    analysisMetrics: {
      accuracy: 0.96,
      precision: 0.94,
      recall: 0.91,
      inferenceSpeed: 45.2,
    },
    technicalDetails: {
      imageSize: "2.4 MB",
      dimensions: "1920×1080",
      format: "JPEG",
      graphNodes: 1542,
      graphEdges: 8924,
    },
  },
  {
    subject: "Mountain Landscape",
    classId: 2,
    category: "Two Class",
    categoryColor: "#4ECDC4",
    confidence: 0.91,
    processingTime: 1100,
    modelVersion: "GNN-v2.1.4",
    topPredictions: [
      { subject: "Mountain Landscape", classId: 2, category: "Two Class", confidence: 0.91, categoryColor: "#4ECDC4" },
      { subject: "Forest Landscape", classId: 5, category: "Five Class", confidence: 0.85, categoryColor: "#98D8C8" },
      { subject: "Mountain Range", classId: 3, category: "Three Class", confidence: 0.82, categoryColor: "#45B7D1" },
    ],
    analysisMetrics: {
      accuracy: 0.95,
      precision: 0.92,
      recall: 0.89,
      inferenceSpeed: 47.1,
    },
    technicalDetails: {
      imageSize: "1.8 MB",
      dimensions: "1920×1080",
      format: "JPEG",
      graphNodes: 1320,
      graphEdges: 7650,
    },
  },
  {
    subject: "Sports Car",
    classId: 4,
    category: "Four Class",
    categoryColor: "#FFA07A",
    confidence: 0.88,
    processingTime: 980,
    modelVersion: "GNN-v2.1.4",
    topPredictions: [
      { subject: "Sports Car", classId: 4, category: "Four Class", confidence: 0.88, categoryColor: "#FFA07A" },
      { subject: "Luxury Car", classId: 6, category: "Six Class", confidence: 0.82, categoryColor: "#F7DC6F" },
      { subject: "Automobile", classId: 7, category: "Seven Class", confidence: 0.85, categoryColor: "#BB8FCE" },
    ],
    analysisMetrics: {
      accuracy: 0.93,
      precision: 0.89,
      recall: 0.86,
      inferenceSpeed: 49.5,
    },
    technicalDetails: {
      imageSize: "2.1 MB",
      dimensions: "1920×1080",
      format: "JPEG",
      graphNodes: 1450,
      graphEdges: 8230,
    },
  },
  {
    subject: "Modern Architecture",
    classId: 3,
    category: "Three Class",
    categoryColor: "#45B7D1",
    confidence: 0.92,
    processingTime: 1350,
    modelVersion: "GNN-v2.1.4",
    topPredictions: [
      { subject: "Modern Architecture", classId: 3, category: "Three Class", confidence: 0.92, categoryColor: "#45B7D1" },
      { subject: "Contemporary Building", classId: 8, category: "Eight Class", confidence: 0.87, categoryColor: "#85C1E2" },
      { subject: "Architectural Design", classId: 1, category: "One Class", confidence: 0.84, categoryColor: "#FF6B6B" },
    ],
    analysisMetrics: {
      accuracy: 0.94,
      precision: 0.91,
      recall: 0.88,
      inferenceSpeed: 43.8,
    },
    technicalDetails: {
      imageSize: "2.3 MB",
      dimensions: "1920×1080",
      format: "JPEG",
      graphNodes: 1620,
      graphEdges: 9100,
    },
  },
  {
    subject: "Sunset Beach",
    classId: 5,
    category: "Five Class",
    categoryColor: "#98D8C8",
    confidence: 0.89,
    processingTime: 1050,
    modelVersion: "GNN-v2.1.4",
    topPredictions: [
      { subject: "Sunset Beach", classId: 5, category: "Five Class", confidence: 0.89, categoryColor: "#98D8C8" },
      { subject: "Beach Sunset", classId: 2, category: "Two Class", confidence: 0.85, categoryColor: "#4ECDC4" },
      { subject: "Ocean Sunset", classId: 9, category: "Nine Class", confidence: 0.82, categoryColor: "#F8B88B" },
    ],
    analysisMetrics: {
      accuracy: 0.92,
      precision: 0.88,
      recall: 0.85,
      inferenceSpeed: 46.3,
    },
    technicalDetails: {
      imageSize: "1.9 MB",
      dimensions: "1920×1080",
      format: "JPEG",
      graphNodes: 1280,
      graphEdges: 7420,
    },
  },
  {
    subject: "Portrait",
    classId: 6,
    category: "Six Class",
    categoryColor: "#F7DC6F",
    confidence: 0.95,
    processingTime: 1420,
    modelVersion: "GNN-v2.1.4",
    topPredictions: [
      { subject: "Portrait", classId: 6, category: "Six Class", confidence: 0.95, categoryColor: "#F7DC6F" },
      { subject: "Human Face", classId: 7, category: "Seven Class", confidence: 0.92, categoryColor: "#BB8FCE" },
      { subject: "Person", classId: 4, category: "Four Class", confidence: 0.94, categoryColor: "#FFA07A" },
    ],
    analysisMetrics: {
      accuracy: 0.97,
      precision: 0.95,
      recall: 0.93,
      inferenceSpeed: 42.1,
    },
    technicalDetails: {
      imageSize: "2.5 MB",
      dimensions: "1920×1080",
      format: "JPEG",
      graphNodes: 1750,
      graphEdges: 9650,
    },
  },
  {
    subject: "Food Dish",
    classId: 7,
    category: "Seven Class",
    categoryColor: "#BB8FCE",
    confidence: 0.87,
    processingTime: 920,
    modelVersion: "GNN-v2.1.4",
    topPredictions: [
      { subject: "Food Dish", classId: 7, category: "Seven Class", confidence: 0.87, categoryColor: "#BB8FCE" },
      { subject: "Cuisine", classId: 1, category: "One Class", confidence: 0.83, categoryColor: "#FF6B6B" },
      { subject: "Meal", classId: 4, category: "Four Class", confidence: 0.85, categoryColor: "#FFA07A" },
    ],
    analysisMetrics: {
      accuracy: 0.91,
      precision: 0.86,
      recall: 0.84,
      inferenceSpeed: 48.7,
    },
    technicalDetails: {
      imageSize: "1.7 MB",
      dimensions: "1920×1080",
      format: "JPEG",
      graphNodes: 1200,
      graphEdges: 6980,
    },
  },
  {
    subject: "City Skyline",
    classId: 8,
    category: "Eight Class",
    categoryColor: "#85C1E2",
    confidence: 0.9,
    processingTime: 1150,
    modelVersion: "GNN-v2.1.4",
    topPredictions: [
      { subject: "City Skyline", classId: 8, category: "Eight Class", confidence: 0.9, categoryColor: "#85C1E2" },
      { subject: "Urban Skyline", classId: 6, category: "Six Class", confidence: 0.86, categoryColor: "#F7DC6F" },
      { subject: "Cityscape", classId: 3, category: "Three Class", confidence: 0.88, categoryColor: "#45B7D1" },
    ],
    analysisMetrics: {
      accuracy: 0.93,
      precision: 0.89,
      recall: 0.87,
      inferenceSpeed: 45.9,
    },
    technicalDetails: {
      imageSize: "2.2 MB",
      dimensions: "1920×1080",
      format: "JPEG",
      graphNodes: 1550,
      graphEdges: 8450,
    },
  },
  {
    subject: "Flower Garden",
    classId: 9,
    category: "Nine Class",
    categoryColor: "#F8B88B",
    confidence: 0.93,
    processingTime: 1280,
    modelVersion: "GNN-v2.1.4",
    topPredictions: [
      { subject: "Flower Garden", classId: 9, category: "Nine Class", confidence: 0.93, categoryColor: "#F8B88B" },
      { subject: "Garden", classId: 5, category: "Five Class", confidence: 0.89, categoryColor: "#98D8C8" },
      { subject: "Flowers", classId: 2, category: "Two Class", confidence: 0.91, categoryColor: "#4ECDC4" },
    ],
    analysisMetrics: {
      accuracy: 0.95,
      precision: 0.92,
      recall: 0.9,
      inferenceSpeed: 44.5,
    },
    technicalDetails: {
      imageSize: "2.0 MB",
      dimensions: "1920×1080",
      format: "JPEG",
      graphNodes: 1480,
      graphEdges: 8120,
    },
  },
]

export async function POST(request: NextRequest) {
  try {
    const { image } = await request.json()

    if (!image) {
      return NextResponse.json({ error: "No image provided" }, { status: 400 })
    }

    // Simulate GNN processing time
    await new Promise((resolve) => setTimeout(resolve, 500))

    // Return a random classification result (in production, this would call your actual GNN model)
    const result = subjects[Math.floor(Math.random() * subjects.length)]

    return NextResponse.json(result)
  } catch (error) {
    console.error("[v0] Classification API error:", error)
    return NextResponse.json({ error: "Classification failed" }, { status: 500 })
  }
}
