'use client'
import React, { useState } from "react";
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
import {
  FileAudio,
  Upload,
  Download,
  AlertCircle,
  BarChart2,
  Activity,
  Waves,
  Zap,
} from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Progress } from "@/components/ui/progress";

export default function AudioSteganography() {
  const [selected, setSelected] = useState("Encode");
  const [audioFile, setAudioFile] = useState(null);
  const [message, setMessage] = useState("");
  const [processedAudio, setProcessedAudio] = useState(null);
  const [decodedMessage, setDecodedMessage] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [encodingMethod, setEncodingMethod] = useState("");

  const API_URL = "http://127.0.0.1:8000";

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!audioFile) {
      setError("Please select an audio file");
      return;
    }

    setIsLoading(true);
    setError(null);
    setProgress(20);

    const formData = new FormData();
    formData.append("file", audioFile);

    try {
      let response;
      if (selected === "Encode") {
        if (!message.trim()) {
          setError("Please enter a message to hide");
          setIsLoading(false);
          return;
        }
        formData.append("message", message);
        setProgress(40);

        response = await fetch(`${API_URL}/upload`, {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`Server error: ${response.status}`);
        }

        // Get encoding method from headers
        const method = response.headers.get("X-Encoding-Method") || "Unknown";
        setEncodingMethod(method);

        setProgress(80);
        const blob = await response.blob();
        const audioUrl = URL.createObjectURL(blob);
        setProcessedAudio(audioUrl);
        setProgress(100);
      } else if (selected === "Decode") {
        setProgress(40);
        response = await fetch(`${API_URL}/decode`, {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`Server error: ${response.status}`);
        }

        setProgress(80);
        const data = await response.json();
        setDecodedMessage(data.decoded_message);
        setProgress(100);
      } else if (selected === "Analyze") {
        setProgress(40);
        response = await fetch(`${API_URL}/analyze`, {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`Server error: ${response.status}`);
        }

        setProgress(80);
        const data = await response.json();
        setAnalysisResults(data.analysis_results);
        setProgress(100);
      }
    } catch (err) {
      setError(err.message || "An error occurred while processing your file");
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileChange = (e) => {
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
    setAnalysisResults(null);
    setProgress(0);
    setError(null);
    setEncodingMethod("");
  };

  const renderAnalysisResults = () => {
    if (!analysisResults) return null;

    const confidenceColor = analysisResults.confidence_color || "green";
    const likelihood = (analysisResults.steg_likelihood * 100).toFixed(1);

    return (
      <CardFooter className="flex flex-col gap-6">
        <div className="w-full pt-4 border-t">
          <h3 className="font-semibold mb-4 flex items-center gap-2 text-lg">
            <BarChart2 className="h-5 w-5" /> Comprehensive Steganalysis Results
          </h3>

          {/* Summary Card */}
          <div className="bg-gray-50 p-4 rounded-lg mb-6">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <p className="text-gray-600">Duration</p>
                <p className="font-medium">
                  {analysisResults.analysis_summary?.duration?.toFixed(2)}s
                </p>
              </div>
              <div>
                <p className="text-gray-600">Sample Rate</p>
                <p className="font-medium">
                  {analysisResults.analysis_summary?.sample_rate} Hz
                </p>
              </div>
              <div>
                <p className="text-gray-600">Samples</p>
                <p className="font-medium">
                  {analysisResults.analysis_summary?.total_samples?.toLocaleString()}
                </p>
              </div>
              <div>
                <p className="text-gray-600">Channels</p>
                <p className="font-medium">
                  {analysisResults.analysis_summary?.channels}
                </p>
              </div>
            </div>
          </div>

          {/* Detection Confidence */}
          <div className="mb-6">
            <h4 className="text-sm font-semibold mb-2 flex items-center gap-2">
              <AlertCircle className="h-4 w-4" />
              Steganography Detection
            </h4>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm">Likelihood Score</span>
                <span
                  className={`text-sm font-medium ${
                    confidenceColor === "green"
                      ? "text-green-600"
                      : confidenceColor === "yellow"
                      ? "text-yellow-600"
                      : "text-red-600"
                  }`}>
                  {likelihood}% ({analysisResults.detection_confidence})
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-3">
                <div
                  className={`h-3 rounded-full transition-all duration-500 ${
                    confidenceColor === "green"
                      ? "bg-green-500"
                      : confidenceColor === "yellow"
                      ? "bg-yellow-500"
                      : "bg-red-500"
                  }`}
                  style={{ width: `${likelihood}%` }}></div>
              </div>
              <p className="text-xs text-gray-600 mt-1">
                {analysisResults.detection_confidence === "Low"
                  ? "No strong indicators of hidden data detected"
                  : analysisResults.detection_confidence === "Medium"
                  ? "Some anomalies detected - possible steganography"
                  : "Strong indicators of hidden data present"}
              </p>
            </div>
          </div>

          {/* Energy Distribution */}
          <div className="mb-6">
            <h4 className="text-sm font-semibold mb-3 flex items-center gap-2">
              <Waves className="h-4 w-4" />
              Frequency Band Energy Distribution
            </h4>
            <div className="grid grid-cols-3 gap-4">
              {[
                {
                  name: "Low Band",
                  value: analysisResults.energy_distribution?.low_band,
                  color: "bg-blue-500",
                },
                {
                  name: "Mid Band",
                  value: analysisResults.energy_distribution?.mid_band,
                  color: "bg-purple-500",
                },
                {
                  name: "High Band",
                  value: analysisResults.energy_distribution?.high_band,
                  color: "bg-pink-500",
                },
              ].map((band, idx) => (
                <div key={idx} className="text-center">
                  <p className="text-xs font-medium mb-2">{band.name}</p>
                  <div className="h-24 bg-gray-100 rounded relative overflow-hidden">
                    <div
                      className={`absolute bottom-0 w-full ${band.color} transition-all duration-700`}
                      style={{ height: `${band.value * 100}%` }}></div>
                  </div>
                  <p className="text-xs mt-2 font-medium">
                    {(band.value * 100).toFixed(1)}%
                  </p>
                </div>
              ))}
            </div>
          </div>

          {/* Spectral Features */}
          <div className="grid md:grid-cols-2 gap-6 mb-6">
            <div>
              <h4 className="text-sm font-semibold mb-3 flex items-center gap-2">
                <Activity className="h-4 w-4" />
                Spectral Analysis
              </h4>
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">
                    Spectral Flatness
                  </span>
                  <span className="text-sm font-medium">
                    {analysisResults.spectral_flatness?.toFixed(4)}
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="h-2 bg-indigo-500 rounded-full transition-all duration-500"
                    style={{
                      width: `${analysisResults.spectral_flatness * 100}%`,
                    }}></div>
                </div>
                <p className="text-xs text-gray-500">
                  {analysisResults.spectral_flatness < 0.3
                    ? "Low flatness may indicate hidden data"
                    : "Normal spectral characteristics"}
                </p>
              </div>
            </div>

            <div>
              <h4 className="text-sm font-semibold mb-3 flex items-center gap-2">
                <Zap className="h-4 w-4" />
                Audio Properties
              </h4>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">
                    Spectral Centroid
                  </span>
                  <span className="text-sm font-medium">
                    {(analysisResults.spectral_centroid / 1000).toFixed(2)} kHz
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">
                    Zero Crossing Rate
                  </span>
                  <span className="text-sm font-medium">
                    {(analysisResults.zero_crossing_rate * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">RMS Energy</span>
                  <span className="text-sm font-medium">
                    {analysisResults.rms_energy?.toFixed(4)}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Anomaly Detection */}
          {analysisResults.frequency_anomalies && (
            <div className="mb-6">
              <h4 className="text-sm font-semibold mb-3 flex items-center gap-2">
                <AlertCircle className="h-4 w-4" />
                Anomaly Detection
              </h4>
              <div className="grid md:grid-cols-2 gap-4">
                <div className="bg-gray-50 p-3 rounded">
                  <p className="text-sm font-medium">
                    High Frequency Variability
                  </p>
                  <p className="text-xs text-gray-600 mt-1">
                    CV:{" "}
                    {analysisResults.frequency_anomalies.high_freq_cv?.toFixed(
                      3
                    )}
                    {analysisResults.frequency_anomalies.high_freq_cv > 0.5
                      ? " (High - Potential LSB)"
                      : " (Normal)"}
                  </p>
                </div>
                <div className="bg-gray-50 p-3 rounded">
                  <p className="text-sm font-medium">Distribution Deviation</p>
                  <p className="text-xs text-gray-600 mt-1">
                    {(
                      analysisResults.frequency_anomalies
                        .distribution_deviation * 100
                    ).toFixed(1)}
                    %
                    {analysisResults.frequency_anomalies
                      .distribution_deviation > 0.3
                      ? " (Anomalous)"
                      : " (Normal)"}
                  </p>
                </div>
              </div>
            </div>
          )}

          <div className="flex gap-2 mt-6">
            <Button variant="secondary" className="flex-1" onClick={resetForm}>
              Analyze Another File
            </Button>
          </div>
        </div>
      </CardFooter>
    );
  };

  return (
    <div className="container mx-auto py-10 px-4">
      <h1 className="text-3xl font-bold text-center mb-8">
        AI-Powered Audio Steganography
      </h1>
      <p className="text-center text-gray-600 mb-8">
        Using Reinforcement Learning for optimal encoding method selection
      </p>

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
              {["Encode", "Decode", "Analyze"].map((option) => (
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
              {selected === "Encode"
                ? "AI-Powered Message Encoding"
                : selected === "Decode"
                ? "Neural Message Decoder"
                : "Advanced Steganalysis"}
            </CardTitle>
            <CardDescription>
              {selected === "Encode"
                ? "Our RL agent automatically selects the optimal encoding method for your audio file!"
                : selected === "Decode"
                ? "Extract hidden messages using our trained neural network decoder."
                : "Comprehensive analysis using machine learning to detect potential steganography."}
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
                  <p className="text-xs text-gray-500">
                    💡 Our RL agent will automatically choose the best encoding
                    method for your audio
                  </p>
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
                    {selected === "Encode"
                      ? "Smart Encode"
                      : selected === "Decode"
                      ? "🔓 AI Decode"
                      : "🔬 Deep Analyze"}{" "}
                    {selected === "Analyze" ? (
                      <BarChart2 className="ml-2 h-4 w-4" />
                    ) : (
                      <Upload className="ml-2 h-4 w-4" />
                    )}
                  </>
                )}
              </Button>
            </form>

            {isLoading && (
              <div className="mt-6 space-y-2">
                <Progress value={progress} className="h-2" />
                <p className="text-sm text-center text-muted-foreground">
                  {progress < 40
                    ? "Initializing..."
                    : progress < 80
                    ? "Processing with AI..."
                    : "Finalizing results..."}
                </p>
              </div>
            )}
          </CardContent>

          {processedAudio && audioFile && selected === "Encode" && (
            <CardFooter className="flex flex-col gap-4">
              {encodingMethod && (
                <div className="w-full pt-2">
                  <div className="bg-green-50 border border-green-200 rounded-lg p-3">
                    <p className="text-sm font-medium text-green-800">
                      🎯 RL Agent Selected Method:{" "}
                      <strong>{encodingMethod}</strong>
                    </p>
                    <p className="text-xs text-green-600 mt-1">
                      Optimized based on your audio characteristics
                    </p>
                  </div>
                </div>
              )}

              <div className="w-full pt-4 border-t">
                <h3 className="font-medium mb-2 flex items-center gap-2">
                  <FileAudio className="h-4 w-4" /> Original Audio
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
                  <FileAudio className="h-4 w-4" /> Steganographic Audio
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
                    Encode Another
                  </Button>
                  <Button
                    variant="default"
                    className="flex-1"
                    onClick={() => {
                      const a = document.createElement("a");
                      a.href = processedAudio;
                      a.download = `stego-${
                        encodingMethod?.toLowerCase() || "encoded"
                      }-${audioFile?.name || "audio.flac"}`;
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
                  🔓 Decoded Secret Message
                </h3>
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
                  <p className="font-mono text-sm break-words">
                    {decodedMessage}
                  </p>
                </div>
                <p className="text-xs text-gray-500">
                  Message length: {decodedMessage.length} characters
                </p>
                <div className="flex gap-2 mt-4">
                  <Button
                    variant="secondary"
                    className="flex-1"
                    onClick={resetForm}>
                    Decode Another
                  </Button>
                  <Button
                    variant="outline"
                    className="flex-1"
                    onClick={() => {
                      navigator.clipboard.writeText(decodedMessage);
                    }}>
                    Copy Message
                  </Button>
                </div>
              </div>
            </CardFooter>
          )}

          {analysisResults && selected === "Analyze" && renderAnalysisResults()}
        </Card>
      </div>

      <div className="max-w-md mx-auto mt-8 text-center text-sm text-muted-foreground">
        <p>
          🧠 Powered by Reinforcement Learning and Neural Networks for
          intelligent audio steganography. Hidden data is imperceptible to human
          ears while maintaining audio quality.
        </p>
      </div>

      <style jsx>{`
        .radio-inputs {
          position: relative;
          display: flex;
          flex-wrap: wrap;
          border-radius: 0.5rem;
          background-color: #eee;
          box-sizing: border-box;
          box-shadow: 0 0 0px 1px rgba(0, 0, 0, 0.06);
          padding: 0.25rem;
          width: 300px;
          font-size: 14px;
          margin: 1rem 0;
        }

        .radio-inputs .radio {
          flex: 1 1 auto;
          text-align: center;
        }

        .radio-inputs .radio input {
          display: none;
        }

        .radio-inputs .radio .name {
          display: flex;
          cursor: pointer;
          align-items: center;
          justify-content: center;
          border-radius: 0.5rem;
          border: none;
          padding: 0.5rem 0;
          color: rgba(51, 65, 85, 1);
          transition: all 0.15s ease-in-out;
        }

        .radio-inputs .radio input:checked + .name {
          background-color: #fff;
          font-weight: 600;
        }
      `}</style>
    </div>
  );
}
