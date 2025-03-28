"use client"

import type React from "react"
import { useState, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Slider } from "@/components/ui/slider"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Upload, ImageIcon, Video, Play, LinkIcon } from "lucide-react"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { AlertCircle } from "lucide-react"
import Image from "next/image"
import { ModelInfo } from "@/app/model-Info";

import { ProcessingIndicator } from "@/components/processing-indicator"

export default function RoadBoundaryDetector() {
  const [selectedModel, setSelectedModel] = useState("daytime")
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [selectedFileType, setSelectedFileType] = useState<"image" | "video" | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [confidenceThreshold, setConfidenceThreshold] = useState(50)
  const [displayMode, setDisplayMode] = useState("draw")
  const [isProcessing, setIsProcessing] = useState(false)
  const [resultUrl, setResultUrl] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const sampleImages = [
    "/samples/frame_0002.jpg",
    "/samples/frame_0003.jpg",
    "/samples/frame_0241.jpg",
    "/samples/frame_0242.jpg",
  ]

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    const fileType = file.type.startsWith("image/") ? "image" : "video"
    setSelectedFileType(fileType)
    setSelectedFile(file)

    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
    setResultUrl(null)
    setError(null)
  }

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()

    const file = e.dataTransfer.files?.[0]
    if (!file) return

    const fileType = file.type.startsWith("image/") ? "image" : "video"
    setSelectedFileType(fileType)
    setSelectedFile(file)

    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
    setResultUrl(null)
    setError(null)
  }

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
  }

  const handleSampleImageClick = (imagePath: string) => {
    setSelectedFileType("image")
    setPreviewUrl(imagePath)
    setSelectedFile(null)
    setResultUrl(null)
    setError(null)
  }

  const handleUrlSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    const formData = new FormData(e.currentTarget)
    const url = formData.get("url") as string

    if (url) {
      setPreviewUrl(url)
      setSelectedFile(null)
      setSelectedFileType("image")
      setResultUrl(null)
      setError(null)
    }
  }

  const processImage = async () => {
    if (!previewUrl) {
      setError("No image or video selected")
      return
    }

    setIsProcessing(true)
    setError(null)

    try {
      const formData = new FormData()

      // Add image to FormData
      if (selectedFile) {
        // Check file size
        if (selectedFile.size > 10 * 1024 * 1024) {
          throw new Error("File size exceeds 10MB limit")
        }
        formData.append("file", selectedFile)
      } else if (previewUrl) {
        try {
          const response = await fetch(previewUrl)
          if (!response.ok) {
            throw new Error(`Failed to fetch image: ${response.status}`)
          }
          const blob = await response.blob()
          formData.append("file", blob, "image.jpg")
        } catch (fetchError: unknown) {
          const errorMessage =
            fetchError instanceof Error ? fetchError.message : "Unknown error occurred while fetching the image"
          throw new Error(`Failed to fetch image: ${errorMessage}`)
        }
      }

      // Add other parameters
      formData.append("model", selectedModel)
      formData.append("confidence", (confidenceThreshold / 100).toString())
      formData.append("displayMode", displayMode)

      // Make the API request
      const response = await fetch("/api/process", {
        method: "POST",
        body: formData,
      })

      // Handle response
      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(
          `Failed to process image: Server returned ${response.status} ${response.statusText}. Details: ${errorText}`,
        )
      }

      // Process successful response
      const data = await response.json()

      if (!data.fileId) {
        throw new Error("Server response missing required fileId")
      }

      setResultUrl(`/api/results/${data.fileId}`)
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : "An unknown error occurred"
      console.error("Error processing image:", errorMessage)
      setError(errorMessage)
    } finally {
      setIsProcessing(false)
    }
  }

  return (
    <div className="container mx-auto py-6 max-w-7xl">
      <h1 className="text-3xl font-bold mb-6">Road Boundary Detector</h1>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="md:col-span-1 space-y-6">
          {/* Model Selection */}
          <ModelInfo selectedModel={selectedModel} setSelectedModel={setSelectedModel} />

          {/* Sample Images */}
          <Card>
            <CardHeader>
              <CardTitle>Samples from Test Set</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-2">
                {sampleImages.map((image, index) => (
                  <div
                    key={index}
                    className="cursor-pointer border rounded-md overflow-hidden hover:opacity-80 transition-opacity"
                    onClick={() => handleSampleImageClick(image)}
                  >
                    <Image
                      src={image || "/placeholder.svg"}
                      alt={`Sample ${index + 1}`}
                      width={150}
                      height={150}
                      className="object-cover aspect-square"
                    />
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Upload Image/Video */}
          <Card>
            <CardHeader>
              <CardTitle>Upload File</CardTitle>
              <CardDescription>Upload an image or video for processing</CardDescription>
            </CardHeader>
            <CardContent>
              <div
                className="border-2 border-dashed rounded-lg p-6 text-center cursor-pointer hover:bg-muted/50 transition-colors"
                onClick={() => fileInputRef.current?.click()}
                onDrop={handleDrop}
                onDragOver={handleDragOver}
              >
                <div className="flex flex-col items-center gap-2">
                  <Upload className="h-10 w-10 text-muted-foreground" />
                  <p className="font-medium">Drop file here or click to browse</p>
                  <p className="text-sm text-muted-foreground">Supports images and videos</p>
                </div>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*,video/*"
                  className="hidden"
                  onChange={handleFileChange}
                />
              </div>
              <div className="flex gap-2 mt-4">
                <Button
                  variant="outline"
                  className="w-1/2"
                  onClick={() => {
                    fileInputRef.current?.click()
                    fileInputRef.current?.setAttribute("accept", "image/*")
                  }}
                >
                  <ImageIcon className="mr-2 h-4 w-4" />
                  Image
                </Button>
                <Button
                  variant="outline"
                  className="w-1/2"
                  onClick={() => {
                    fileInputRef.current?.click()
                    fileInputRef.current?.setAttribute("accept", "video/*")
                  }}
                >
                  <Video className="mr-2 h-4 w-4" />
                  Video
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Paste URL */}
          <Card>
            <CardHeader>
              <CardTitle>Paste Image URL</CardTitle>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleUrlSubmit} className="flex gap-2">
                <div className="relative flex-1">
                  <LinkIcon className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
                  <input
                    type="url"
                    name="url"
                    placeholder="Paste a link..."
                    className="pl-8 h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                  />
                </div>
                <Button type="submit" variant="outline">
                  Submit
                </Button>
              </form>
            </CardContent>
          </Card>

          {/* Run Button */}
          <Button className="w-full py-6 text-lg" disabled={!previewUrl || isProcessing} onClick={processImage}>
            {isProcessing ? "Processing..." : "Run Results"}
            {!isProcessing && <Play className="ml-2 h-4 w-4" />}
          </Button>
        </div>

        {/* Main Content Area */}
        <div className="md:col-span-2 space-y-6">
          <Card className="overflow-hidden">
            <CardContent className="p-0 relative">
              {resultUrl ? (
                selectedFileType === "video" ? (
                  <video src={resultUrl} controls className="w-full h-auto max-h-[600px] object-contain bg-black" />
                ) : (
                  <div className="relative">
                    <Image
                      src={resultUrl || "/placeholder.svg"}
                      alt="Result"
                      width={1200}
                      height={800}
                      className="w-full h-auto max-h-[600px] object-contain bg-black"
                    />
                    <div className="absolute bottom-4 left-4 bg-yellow-400/80 text-black px-2 py-1 rounded">
                      Road-Boundary {confidenceThreshold}%
                    </div>
                  </div>
                )
              ) : previewUrl ? (
                selectedFileType === "video" ? (
                  <video src={previewUrl} controls className="w-full h-auto max-h-[600px] object-contain bg-black" />
                ) : (
                  <Image
                    src={previewUrl || "/placeholder.svg"}
                    alt="Preview"
                    width={1200}
                    height={800}
                    className="w-full h-auto max-h-[600px] object-contain bg-black"
                  />
                )
              ) : (
                <div className="flex items-center justify-center h-[600px] bg-muted/20">
                  <p className="text-muted-foreground">Upload an image or video to see preview</p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Processing Indicator */}
          {isProcessing && (
            <div className="my-4">
              <ProcessingIndicator isProcessing={isProcessing} />
            </div>
          )}

          {/* Error Display */}
          {error && (
            <Alert variant="destructive" className="my-4">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Error</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {/* Settings */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Confidence Threshold</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex justify-between">
                    <span>0%</span>
                    <span className="font-medium">{confidenceThreshold}%</span>
                    <span>100%</span>
                  </div>
                  <Slider
                    value={[confidenceThreshold]}
                    min={0}
                    max={100}
                    step={1}
                    onValueChange={(value) => setConfidenceThreshold(value[0])}
                  />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Label Display Mode</CardTitle>
              </CardHeader>
              <CardContent>
                <Select value={displayMode} onValueChange={setDisplayMode}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select display mode" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="draw">Draw Labels</SelectItem>
                    <SelectItem value="highlight">Highlight Boundaries</SelectItem>
                    <SelectItem value="outline">Outline Only</SelectItem>
                    <SelectItem value="none">No Labels</SelectItem>
                  </SelectContent>
                </Select>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}

