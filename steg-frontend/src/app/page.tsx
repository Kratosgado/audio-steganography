"use client";

import type React from "react";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { FileAudio, Upload, Download, AlertCircle } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Progress } from "@/components/ui/progress";

export default function AudioSteganography() {
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [message, setMessage] = useState("");
  const [processedAudio, setProcessedAudio] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (file.type.startsWith("audio/")) {
        setAudioFile(file);
        setError(null);
      } else {
        setError("Please upload an audio file");
        setAudioFile(null);
      }
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!audioFile) {
      setError("Please select an audio file");
      return;
    }

    if (!message.trim()) {
      setError("Please enter a message to hide");
      return;
    }

    setIsLoading(true);
    setError(null);

    // Simulate processing with progress
    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 95) {
          clearInterval(interval);
          return prev;
        }
        return prev + 5;
      });
    }, 200);

    try {
      // In a real application, you would send the file and message to your backend
      // Here we're simulating the process
      await new Promise((resolve) => setTimeout(resolve, 2000));

      // Create a mock URL for the processed audio
      // In a real app, this would come from your backend
      const mockProcessedAudioUrl = URL.createObjectURL(audioFile);
      setProcessedAudio(mockProcessedAudioUrl);

      clearInterval(interval);
      setProgress(100);
    } catch {
      setError("An error occurred while processing your file");
      clearInterval(interval);
    } finally {
      setIsLoading(false);
    }
  };

  const resetForm = () => {
    setAudioFile(null);
    setMessage("");
    setProcessedAudio(null);
    setProgress(0);
    setError(null);
  };

  return (
    <div className="container mx-auto py-10 px-4">
      <h1 className="text-3xl font-bold text-center mb-8">
        Audio Steganography
      </h1>

      <div className="max-w-md mx-auto">
        {error && (
          <Alert variant="destructive" className="mb-6">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <Card>
          <CardHeader>
            <CardTitle>Hide Data in Audio</CardTitle>
            <CardDescription>
              Upload an audio file and enter a secret message to hide within the
              audio.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="space-y-2">
                <Label htmlFor="audio-file">Audio File</Label>
                <div className="flex items-center gap-2">
                  <Input
                    id="audio-file"
                    type="file"
                    accept="audio/*"
                    onChange={handleFileChange}
                    disabled={isLoading}
                    className="flex-1"
                  />
                </div>
                {audioFile && (
                  <p className="text-sm text-muted-foreground">
                    Selected: {audioFile.name} (
                    {(audioFile.size / 1024).toFixed(2)} KB)
                  </p>
                )}
              </div>

              <div className="space-y-2">
                <Label htmlFor="message">Secret Message</Label>
                <Textarea
                  id="message"
                  placeholder="Enter the message you want to hide..."
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  disabled={isLoading}
                  className="min-h-[100px]"
                />
              </div>

              <Button
                type="submit"
                className="w-full"
                disabled={isLoading || !audioFile}>
                {isLoading ? (
                  <>
                    Processing...{" "}
                    <Upload className="ml-2 h-4 w-4 animate-pulse" />
                  </>
                ) : (
                  <>
                    Hide Message <Upload className="ml-2 h-4 w-4" />
                  </>
                )}
              </Button>
            </form>

            {isLoading && (
              <div className="mt-6 space-y-2">
                <Progress value={progress} className="h-2" />
                <p className="text-sm text-center text-muted-foreground">
                  Processing your audio file...
                </p>
              </div>
            )}
          </CardContent>

          {processedAudio && (
            <CardFooter className="flex flex-col gap-4">
              <div className="w-full pt-4 border-t">
                <h3 className="font-medium mb-2 flex items-center gap-2">
                  <FileAudio className="h-4 w-4" /> Processed Audio
                </h3>
                <audio controls className="w-full mb-4">
                  <source src={processedAudio} type={audioFile?.type} />
                  Your browser does not support the audio element.
                </audio>
                <div className="flex gap-2">
                  <Button
                    variant="secondary"
                    className="flex-1"
                    onClick={resetForm}>
                    Start Over
                  </Button>
                  <Button
                    variant="default"
                    className="flex-1"
                    onClick={() => {
                      const a = document.createElement("a");
                      a.href = processedAudio;
                      a.download = `stego-${audioFile?.name || "audio.mp3"}`;
                      document.body.appendChild(a);
                      a.click();
                      document.body.removeChild(a);
                    }}>
                    Download <Download className="ml-2 h-4 w-4" />
                  </Button>
                </div>
              </div>
            </CardFooter>
          )}
        </Card>
      </div>

      <div className="max-w-md mx-auto mt-8 text-center text-sm text-muted-foreground">
        <p>
          Audio steganography is the practice of hiding data within audio files.
          The hidden data is typically undetectable to human ears.
        </p>
      </div>
    </div>
  );
}

