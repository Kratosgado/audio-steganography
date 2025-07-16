"use client";
import React, { ChangeEvent, FormEvent, useState } from "react";
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
  Shield,
  Brain,
  FileText,
} from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Progress } from "@/components/ui/progress";

export default function AudioSteganography() {
  const [selected, setSelected] = useState("Encode");
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [message, setMessage] = useState("");
  const [processedAudio, setProcessedAudio] = useState<string | null>(null);
  const [decodedMessage, setDecodedMessage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [analysisResults, setAnalysisResults] = useState<any>();
  const [originalMessage] = useState("");
  const [encodingMethod, setEncodingMethod] = useState("");
  const [encodedAudioBlob, setEncodedAudioBlob] = useState<Blob | null>(null);
  const [encodedFileName, setEncodedFileName] = useState("");

  const API_URL = "http://127.0.0.1:8000";

  const handleSubmit = async (e: FormEvent) => {
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
        setEncodedAudioBlob(blob);
        setEncodedFileName(
          `stego-${method?.toLowerCase() || "encoded"}-${
            audioFile?.name || "audio.flac"
          }`,
        );
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
        setEncodingMethod(data.decoding_method || "RL-Enhanced LSB");
        setProgress(100);
      } else if (selected === "Analyze") {
        // Add original message for comparison if provided
        if (originalMessage.trim()) {
          formData.append("original_message", originalMessage.trim());
        }

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
    } catch (err: any) {
      setError(err.message || "An error occurred while processing your file");
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
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
    setAnalysisResults(undefined);
    setProgress(0);
    setError(null);
    setEncodingMethod("");
    setEncodedAudioBlob(null);
    setEncodedFileName("");
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
                  }`}
                >
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
                  style={{ width: `${likelihood}%` }}
                ></div>
              </div>
              <p className="text-xs text-gray-600 mt-1">
                {analysisResults.detection_confidence === "Very Low" ||
                analysisResults.detection_confidence === "Low"
                  ? "No strong indicators of hidden data detected"
                  : analysisResults.detection_confidence === "Medium"
                    ? "Some anomalies detected - possible steganography"
                    : "Strong indicators of hidden data present"}
              </p>
              {analysisResults.analysis_version && (
                <p className="text-xs text-blue-600 mt-1">
                  🔬 Enhanced Analysis v{analysisResults.analysis_version}
                </p>
              )}
            </div>
          </div>

          {/* Enhanced Detection Results */}
          {analysisResults.enhanced_detection && (
            <div className="mb-6">
              <h4 className="text-sm font-semibold mb-3 flex items-center gap-2">
                <Shield className="h-4 w-4" />
                Advanced Detection Analysis
              </h4>
              <div className="grid grid-cols-2 gap-4 text-xs">
                {analysisResults.enhanced_detection.lsb_analysis && (
                  <div className="bg-blue-50 p-3 rounded">
                    <p className="font-medium text-blue-800">LSB Analysis</p>
                    <p className="text-blue-600">
                      Anomaly Score:{" "}
                      {(
                        analysisResults.enhanced_detection.lsb_analysis
                          .lsb_anomaly_score * 100
                      ).toFixed(1)}
                      %
                    </p>
                    <p className="text-blue-600">
                      Chi-Square:{" "}
                      {analysisResults.enhanced_detection.lsb_analysis.chi_square?.toFixed(
                        2,
                      )}
                    </p>
                  </div>
                )}
                {analysisResults.enhanced_detection.frequency_analysis && (
                  <div className="bg-purple-50 p-3 rounded">
                    <p className="font-medium text-purple-800">
                      Frequency Analysis
                    </p>
                    <p className="text-purple-600">
                      Anomaly Score:{" "}
                      {(
                        analysisResults.enhanced_detection.frequency_analysis
                          .freq_anomaly_score * 100
                      ).toFixed(1)}
                      %
                    </p>
                    <p className="text-purple-600">
                      High Freq Ratio:{" "}
                      {(
                        analysisResults.enhanced_detection.frequency_analysis
                          .high_freq_ratio * 100
                      ).toFixed(1)}
                      %
                    </p>
                  </div>
                )}
                {analysisResults.enhanced_detection.statistical_analysis && (
                  <div className="bg-green-50 p-3 rounded">
                    <p className="font-medium text-green-800">
                      Statistical Analysis
                    </p>
                    <p className="text-green-600">
                      Anomaly Score:{" "}
                      {(
                        analysisResults.enhanced_detection.statistical_analysis
                          .stat_anomaly_score * 100
                      ).toFixed(1)}
                      %
                    </p>
                    <p className="text-green-600">
                      Skewness:{" "}
                      {analysisResults.enhanced_detection.statistical_analysis.skewness?.toFixed(
                        3,
                      )}
                    </p>
                  </div>
                )}
                {analysisResults.enhanced_detection.entropy_analysis && (
                  <div className="bg-orange-50 p-3 rounded">
                    <p className="font-medium text-orange-800">
                      Entropy Analysis
                    </p>
                    <p className="text-orange-600">
                      Anomaly Score:{" "}
                      {(
                        analysisResults.enhanced_detection.entropy_analysis
                          .entropy_anomaly_score * 100
                      ).toFixed(1)}
                      %
                    </p>
                    <p className="text-orange-600">
                      Global Entropy:{" "}
                      {analysisResults.enhanced_detection.entropy_analysis.global_entropy?.toFixed(
                        2,
                      )}
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* RL Assessment */}
          {analysisResults.rl_assessment && (
            <div className="mb-6">
              <h4 className="text-sm font-semibold mb-2 flex items-center gap-2">
                <Brain className="h-4 w-4" />
                AI/RL Assessment
              </h4>
              <div className="bg-indigo-50 p-3 rounded-lg">
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <p className="text-indigo-600">Recommended Method:</p>
                    <p className="font-medium text-indigo-800">
                      {analysisResults.rl_assessment.recommended_method}
                    </p>
                  </div>
                  <div>
                    <p className="text-indigo-600">RL Confidence:</p>
                    <p className="font-medium text-indigo-800">
                      {analysisResults.rl_assessment.confidence}
                    </p>
                  </div>
                  <div>
                    <p className="text-indigo-600">Feature Variance:</p>
                    <p className="font-medium text-indigo-800">
                      {analysisResults.rl_assessment.feature_variance?.toFixed(
                        4,
                      )}
                    </p>
                  </div>
                  <div>
                    <p className="text-indigo-600">RL Likelihood:</p>
                    <p className="font-medium text-indigo-800">
                      {(
                        analysisResults.rl_assessment.rl_likelihood * 100
                      ).toFixed(1)}
                      %
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* File Metadata */}
          {analysisResults.metadata_analysis && (
            <div className="mb-6">
              <h4 className="text-sm font-semibold mb-2 flex items-center gap-2">
                <FileText className="h-4 w-4" />
                File Metadata Analysis
              </h4>
              <div className="bg-gray-50 p-3 rounded-lg">
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <p className="text-gray-600">File Size:</p>
                    <p className="font-medium">
                      {analysisResults.metadata_analysis.file_size} bytes
                    </p>
                  </div>
                  <div>
                    <p className="text-gray-600">File Type:</p>
                    <p className="font-medium">
                      {analysisResults.metadata_analysis.file_type}
                    </p>
                  </div>
                  <div>
                    <p className="text-gray-600">Bit Depth:</p>
                    <p className="font-medium">
                      {analysisResults.metadata_analysis.bit_depth} bits
                    </p>
                  </div>
                  <div>
                    <p className="text-gray-600">Encoding:</p>
                    <p className="font-medium">
                      {analysisResults.metadata_analysis.encoding || "Unknown"}
                    </p>
                  </div>
                  {analysisResults.metadata_analysis.creation_time && (
                    <div className="col-span-2">
                      <p className="text-gray-600">Creation Time:</p>
                      <p className="font-medium">
                        {analysisResults.metadata_analysis.creation_time}
                      </p>
                    </div>
                  )}
                  {analysisResults.metadata_analysis.suspicious_metadata && (
                    <div className="col-span-2">
                      <p className="text-red-600 font-medium">
                        ⚠️ Suspicious metadata detected
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

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
                      style={{ height: `${band.value * 100}%` }}
                    ></div>
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
                    }}
                  ></div>
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
                      3,
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
                    onChange={() => {
                      setSelected(option);
                      // Auto-populate decode tab with encoded audio if available
                      if (
                        option === "Decode" &&
                        encodedAudioBlob &&
                        encodedFileName
                      ) {
                        const file = new File(
                          [encodedAudioBlob],
                          encodedFileName,
                          {
                            type: encodedAudioBlob.type || "audio/flac",
                          },
                        );
                        setAudioFile(file);
                        setError(null);
                      }
                    }}
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
                  <div className="space-y-1">
                    <p className="text-sm text-muted-foreground">
                      Selected: {audioFile.name} (
                      {(audioFile.size / 1024).toFixed(2)} KB)
                    </p>
                    {selected === "Decode" &&
                      encodedAudioBlob &&
                      audioFile.name === encodedFileName && (
                        <p className="text-xs text-blue-600 bg-blue-50 px-2 py-1 rounded">
                          🎯 Auto-loaded from previous encoding
                        </p>
                      )}
                  </div>
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
                disabled={isLoading || !audioFile}
              >
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
                    onClick={resetForm}
                  >
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
                    }}
                  >
                    Download <Download className="ml-2 h-4 w-4" />
                  </Button>
                </div>
              </div>
            </CardFooter>
          )}

          {decodedMessage && selected === "Decode" && (
            <CardFooter className="flex flex-col gap-4">
              {encodingMethod && (
                <div className="w-full pt-2">
                  <div className="bg-green-50 border border-green-200 rounded-lg p-3">
                    <p className="text-sm font-medium text-green-800">
                      🤖 RL Agent Decoding Method:{" "}
                      <strong>{encodingMethod}</strong>
                    </p>
                    <p className="text-xs text-green-600 mt-1">
                      Using trained reinforcement learning for optimal decoding
                    </p>
                  </div>
                </div>
              )}

              <div className="w-full pt-4 border-t">
                <h3 className="font-medium mb-2 flex items-center gap-2">
                  🔓 Decoded Secret Message
                </h3>
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
                  <p className="font-mono text-sm break-words">
                    {decodedMessage}
                  </p>
                </div>
                <div className="grid grid-cols-2 gap-4 text-sm mb-4">
                  <div>
                    <p className="text-gray-600">Message Length</p>
                    <p className="font-medium">
                      {decodedMessage.length} characters
                    </p>
                  </div>
                  <div>
                    <p className="text-gray-600">Decoding Success</p>
                    <p className="font-medium text-green-600">✅ Complete</p>
                  </div>
                </div>
                <div className="flex gap-2 mt-4">
                  <Button
                    variant="secondary"
                    className="flex-1"
                    onClick={resetForm}
                  >
                    Decode Another
                  </Button>
                  <Button
                    variant="outline"
                    className="flex-1"
                    onClick={() => {
                      navigator.clipboard.writeText(decodedMessage);
                    }}
                  >
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
