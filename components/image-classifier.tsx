"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Spinner } from "@/components/ui/spinner"
import { Badge } from "@/components/ui/badge"
import { Upload, X, ImageIcon, BarChart3, Clock, Target } from 'lucide-react'
import { classifyImage } from "@/lib/api/model"
import type { ClassificationResult } from "@/lib/types/classification"
import { mapModelToClassificationResult } from "@/lib/api/model.mapper"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

export function ImageClassifier({ isAuthenticated, onNotification, resetSignal }: { isAuthenticated: boolean; onNotification: (msg: string, type: 'success' | 'error' | 'info') => void; resetSignal?: number }) {
  const [selectedImage, setSelectedImage] = useState<string | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [result, setResult] = useState<ClassificationResult | null>(null)
  const [uploadTime, setUploadTime] = useState<number>(0)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const audioRef = useRef<HTMLAudioElement>(null)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [selectedModel, setSelectedModel] = useState<"kNN-Voting" | "GraphSAGE-I_v2" | "GraphSAGE-E_kNN">("kNN-Voting")


  useEffect(() => {
    if (typeof resetSignal !== 'undefined') {
      // clear local classifier state on logout/reset signal
      setSelectedImage(null)
      setIsProcessing(false)
      setResult(null)
      setUploadTime(0)
      if (fileInputRef.current) {
        try { fileInputRef.current.value = "" } catch (e) { /* ignore readonly */ }
      }
    }
  }, [resetSignal])

  const handleImageSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return
  
    const validTypes = ['image/jpeg', 'image/png', 'image/webp']
    if (!validTypes.includes(file.type)) {
      onNotification("Only JPG, PNG, and WebP formats are supported", "error")
      return
    }
  
    if (file.size > 10 * 1024 * 1024) {
      onNotification("File size must be less than 10MB", "error")
      return
    }
  
    // ✅ store the File for backend upload
    setSelectedFile(file)
  
    // 👇 base64 ONLY for preview
    const startTime = Date.now()
    const reader = new FileReader()
    reader.onload = (e) => {
      setUploadTime(Date.now() - startTime)
      setSelectedImage(e.target?.result as string)
      setResult(null)
      onNotification("Image uploaded successfully", "success")
    }
    reader.readAsDataURL(file)
  }
  

  const playNotificationSound = () => {
    if (audioRef.current) {
      audioRef.current.currentTime = 0
      audioRef.current.play().catch(() => {
        console.log("[v0] Audio playback failed or blocked by browser")
      })
    }
  }

  const handleClassify = async () => {
    if (!selectedFile) return
  
    setIsProcessing(true)
    setResult(null)
  
    try {
      const raw = await classifyImage({
        file: selectedFile, // ✅ real File
        model: selectedModel // ✅ add this
      })
  
      const mapped = mapModelToClassificationResult(raw)
      setResult(mapped)
  
      playNotificationSound()
      onNotification("Classification completed successfully", "success")
    } catch (error) {
      console.error("[v0] Classification error:", error)
      onNotification("Classification failed", "error")
    } finally {
      setIsProcessing(false)
    }
  }  

  const handleReset = () => {
    setSelectedImage(null)
    setResult(null)
    setUploadTime(0)
    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }
    onNotification("Image cleared", "info")
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.9) return "text-green-600"
    if (confidence >= 0.7) return "text-yellow-600"
    return "text-red-600"
  }

  const handleUploadClick = () => {
    if (!isAuthenticated) {
      onNotification("Please sign in to upload images", "info")
      return
    }
    fileInputRef.current?.click()
  }

  return (
    <div className="space-y-6">
      <audio
        ref={audioRef}
        src="data:audio/wav;base64,UklGRiYAAABXQVZFZm10IBAAAAABAAEAQB8AAAB9AAACABAAZGF0YQIAAAAAAA=="
      />

      <Card className="p-8">
        {!selectedImage ? (
          <div
            onClick={handleUploadClick}
            className="border-2 border-dashed border-border rounded-lg p-12 text-center cursor-pointer hover:border-primary/50 hover:bg-accent/5 hover:border-solid transition-all"
          >
            <div className="flex flex-col items-center gap-4">
              <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center">
                <Upload className="w-8 h-8 text-primary" />
              </div>
              <div>
                <p className="text-lg font-medium mb-1 text-foreground">Upload an image</p>
                <p className="text-sm text-muted-foreground">Click to browse or drag and drop your image here</p>
              </div>
              <p className="text-xs text-muted-foreground">Supports JPG, PNG, WebP (max 10MB)</p>
            </div>
            <input ref={fileInputRef} type="file" accept="image/*" onChange={handleImageSelect} className="hidden" />
          </div>
        ) : (
          <div className="space-y-6">
            <div className="relative">
              <div className="relative rounded-lg overflow-hidden bg-muted">
                <img
                  src={selectedImage || "/placeholder.svg"}
                  alt="Selected"
                  className="w-full h-auto max-h-96 object-contain mx-auto"
                />
              </div>
              <Button variant="secondary" size="icon" className="absolute top-2 right-2" onClick={handleReset}>
                <X className="w-4 h-4" />
              </Button>
            </div>

            {uploadTime > 0 && (
              <div className="grid grid-cols-2 gap-4 text-center">
                <div className="p-3 rounded-lg bg-muted/50">
                  <p className="text-sm text-muted-foreground">Upload Time</p>
                  <p className="font-semibold text-foreground">{uploadTime}ms</p>
                </div>
                <div className="p-3 rounded-lg bg-muted/50">
                  <p className="text-sm text-muted-foreground">Status</p>
                  <Badge variant="outline" className="bg-green-50 text-green-700">
                    Ready
                  </Badge>
                </div>
              </div>
            )}

            {!result && !isProcessing && (
              <div className="flex gap-4 items-stretch">
                <Button onClick={handleClassify} className="flex-1" size="lg">
                  <ImageIcon className="w-4 h-4 mr-2" />
                  Classify Image
                </Button>

                <Select
                  value={selectedModel}
                  onValueChange={(v) => setSelectedModel(v as any)}
                >
                  <SelectTrigger size="md" className="w-[190px]">
                    <SelectValue placeholder="Choose model" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="kNN-Voting">kNN-Voting</SelectItem>
                    <SelectItem value="GraphSAGE-I_v2">GraphSAGE-I_v2</SelectItem>
                    <SelectItem value="GraphSAGE-E_kNN">GraphSAGE-E_kNN</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            )}

            {isProcessing && (
              <div className="space-y-6 py-6">
                <div className="flex flex-col items-center gap-4">
                  <Spinner className="w-8 h-8" />
                  <div className="text-center">
                    <p className="font-medium text-foreground">Processing image with chosen model</p>
                    <p className="text-sm text-muted-foreground">Analyzing graph structures and patterns</p>
                  </div>
                </div>
              </div>
            )}

            {result && (
              <div className="space-y-6">
                <div
                  className="p-6 rounded-lg border-2"
                  style={{
                    backgroundColor: `${result.categoryColor}15`,
                    borderColor: result.categoryColor,
                  }}
                >
                  <div className="flex items-start justify-between mb-4">
                    <div>
                      <p className="text-sm text-muted-foreground mb-1">Classification Result</p>
                      <h3 className="text-2xl font-bold" style={{ color: result.categoryColor }}>
                        {result.subject}
                      </h3>
                    </div>
                  </div>
                  <div className="flex items-center gap-2 mb-3">
                    <Badge className="text-lg px-3 py-1 text-white" style={{ backgroundColor: result.categoryColor }}>
                      {result.category}
                    </Badge>
                    <span className="text-sm text-muted-foreground">Model: {result.modelVersion}</span>
                  </div>
                  <div className="w-full bg-white-100 rounded-full h-2 dark:bg-gray-700">
                    <div
                      className="h-2 rounded-full"
                      style={{
                        backgroundColor: result.categoryColor,
                        width: `${result.confidence * 100}%`,
                      }}
                    ></div>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <Card className="p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <Clock className="w-4 h-4 text-blue-500" />
                      <span className="font-medium text-foreground">Processing Time</span>
                    </div>
                    <p className="text-2xl font-bold text-foreground">{result.processingTime}ms</p>
                  </Card>

                  <Card className="p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <Target className="w-4 h-4 text-purple-500" />
                      <span className="font-medium text-foreground">Model Confidence</span>
                    </div>
                    <p className="text-2xl font-bold text-foreground">{(result.analysisMetrics.accuracy * 100).toFixed(1)}%</p>
                  </Card>
                </div>

                <Card className="p-6">
                  <div className="flex items-center gap-2 mb-4">
                    <BarChart3 className="w-5 h-5" />
                    <h4 className="font-semibold text-foreground">Top Predictions</h4>
                  </div>
                  <div className="space-y-3">
                    {result.topPredictions.map((prediction, index) => (
                      <div
                        key={index}
                        className="flex items-center justify-between p-3 rounded-lg"
                        style={{
                          backgroundColor: `${prediction.categoryColor}15`,
                          borderLeft: `3px solid ${prediction.categoryColor}`,
                        }}
                      >
                        <div className="flex items-center gap-3">
                          <span className="text-sm font-medium w-6 text-foreground">{index + 1}.</span>
                          <div>
                            <p className="font-medium text-foreground">{prediction.subject}</p>
                            <p className="text-xs text-muted-foreground">{prediction.category}</p>
                          </div>
                        </div>
                        <div className="text-right">
                          <p 
                            className="font-semibold"
                            style={{ color: prediction.categoryColor }}
                          >
                            {(prediction.confidence * 100).toFixed(1)}%
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                </Card>

                <Card className="p-6">
                  <div className="flex items-center gap-2 mb-4">
                    <BarChart3 className="w-5 h-5" />
                    <h4 className="font-semibold text-foreground">Technical Details</h4>
                  </div>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="flex justify-between">
                      <span className="text-foreground">Image Size:</span>
                      <span className="font-medium text-foreground">{result.technicalDetails.imageSize}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-foreground">Dimensions:</span>
                      <span className="font-medium text-foreground">{result.technicalDetails.dimensions}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-foreground">Format:</span>
                      <span className="font-medium text-foreground">{result.technicalDetails.format}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-foreground">Graph Nodes:</span>
                      <span className="font-medium text-foreground">{result.technicalDetails.graphNodes}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-foreground">Graph Edges:</span>
                      <span className="font-medium text-foreground">{result.technicalDetails.graphEdges ?? "None"}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-foreground">Model Confidence:</span>
                      <span className="font-medium text-foreground">{(result.analysisMetrics.accuracy * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                </Card>

                <div className="flex gap-3 items-stretch">
                  {/* New Image */}
                  <Button
                    onClick={handleReset}
                    variant="outline"
                    className="flex-1 bg-transparent"
                  >
                    <Upload className="w-4 h-4 mr-2" />
                    New Image
                  </Button>

                  {/* Reclassify */}
                  <Button
                    onClick={handleClassify}
                    variant="outline"
                    className="flex-1 bg-transparent"
                    disabled={isProcessing}
                  >
                    <ImageIcon className="w-4 h-4 mr-2" />
                    Reclassify
                  </Button>

                  {/* Model Picker */}
                  <Select
                    value={selectedModel}
                    onValueChange={(v: "kNN-Voting" | "GraphSAGE-I_v2" | "GraphSAGE-E_kNN") =>
                      setSelectedModel(v)
                    }
                    disabled={isProcessing}
                  >
                    <SelectTrigger size="md" className="w-[190px]">
                      <SelectValue placeholder="Choose model" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="kNN-Voting">kNN-Voting</SelectItem>
                      <SelectItem value="GraphSAGE-I_v2">GraphSAGE-I_v2</SelectItem>
                      <SelectItem value="GraphSAGE-E_kNN">GraphSAGE-E_kNN</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            )}
          </div>
        )}
      </Card>
    </div>
  )
}
