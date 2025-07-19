import React from "react";
import { Button } from "@/components/ui/button";
import { CardFooter } from "@/components/ui/card";
import {
  BarChart2,
  AlertCircle,
  Shield,
  Brain,
  FileText,
  Waves,
  Activity,
  Zap,
} from "lucide-react";

interface AnalysisResultsProps {
  analysisResults: any;
  onReset: () => void;
}

export function AnalysisResults({ analysisResults, onReset }: AnalysisResultsProps) {
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

        <div className="flex gap-2 mt-6">
          <Button variant="secondary" className="flex-1" onClick={onReset}>
            Analyze Another File
          </Button>
        </div>
      </div>
    </CardFooter>
  );
} 