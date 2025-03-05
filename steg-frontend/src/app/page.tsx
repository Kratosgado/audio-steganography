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
  const [selected, setSelected] = useState<string>("Encode"); // Default selection
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [message, setMessage] = useState("");
  const [processedAudio, setProcessedAudio] = useState<string | null>(null);
  const [decodedMessage, setDecodedMessage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const API_URL = "http://127.0.0.1:8000"; // Backend's base URL

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!audioFile) {
      setError("Please select an audio file");
      return;
    }

    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", audioFile);

    try {
      let response;
      if (selected === "Encode") {
        if (!message.trim()) {
          setError("Please enter a message to hide");
          return;
        }
        formData.append("message", message);
        response = await fetch(`${API_URL}/upload`, {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`Server error: ${response.status}`);
        }

        // Handle the .flac file response
        const blob = await response.blob();
        const audioUrl = URL.createObjectURL(blob);
        setProcessedAudio(audioUrl);
        setProgress(100);
      } else if (selected === "Decode") {
        response = await fetch(`${API_URL}/decode`, {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`Server error: ${response.status}`);
        }

        // Handle the decoded message response
        const data = await response.json();
        setDecodedMessage(data.decoded_message);
        setProgress(100);
      }
    } catch (err: any) {
      setError(err.message || "An error occurred while processing your file");
    } finally {
      setIsLoading(false);
    }
  };

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

  const resetForm = () => {
    setAudioFile(null);
    setMessage("");
    setProcessedAudio(null);
    setDecodedMessage(null);
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
          <div style={{ display: "flex", justifyContent: "center" }}>
            <div className="radio-inputs">
              {["Encode", "Decode"].map((option) => (
                <label key={option} className="radio">
                  <input
                    type="radio"
                    name="radio"
                    value={option}
                    checked={selected === option}
                    onChange={() => setSelected(option)}
                  />
                  <span className="name">{option}</span>
                </label>
              ))}
            </div>
          </div>
          <CardHeader>
            <CardTitle>
              {selected === "Encode" ? "Hide Data in Audio" : "Decode Data from Audio"}
            </CardTitle>
            <CardDescription>
              {selected === "Encode"
                ? "Upload an audio file and enter a secret message to hide within the audio!"
                : "Upload a processed audio file to extract the hidden message."}
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

              {selected === "Encode" && (
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
              )}

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
                    {selected === "Encode" ? "Hide Message" : "Decode Message"}{" "}
                    <Upload className="ml-2 h-4 w-4" />
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

          {processedAudio && selected === "Encode" && (
            <CardFooter className="flex flex-col gap-4">
              <div className="w-full pt-4 border-t">
                <h3 className="font-medium mb-2 flex items-center gap-2">
                  <FileAudio className="h-4 w-4" /> Unprocessed Audio
                </h3>
                <audio controls className="w-full mb-4">
                  <source
                    src={URL.createObjectURL(audioFile)}
                    type={audioFile?.type}
                  />
                  Your browser does not support the audio element.
                </audio>
              </div>
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

          {decodedMessage && selected === "Decode" && (
            <CardFooter className="flex flex-col gap-4">
              <div className="w-full pt-4 border-t">
                <h3 className="font-medium mb-2 flex items-center gap-2">
                  Decoded Message
                </h3>
                <p className="text-sm text-muted-foreground">{decodedMessage}</p>
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