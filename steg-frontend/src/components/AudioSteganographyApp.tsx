"use client";
import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from "@/components/ui/card";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Progress } from "@/components/ui/progress";
import { AlertCircle, Upload, BarChart2 } from "lucide-react";
import { ModeSelector } from "./ModeSelector";
import { AudioFileInput } from "./AudioFileInput";
import { MessageInput } from "./MessageInput";
import { EncodeResults } from "./EncodeResults";
import { DecodeResults } from "./DecodeResults";
import { AnalysisResults } from "./AnalysisResults";

export default function AudioSteganographyApp() {
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
	const [audioAnalysis, setAudioAnalysis] = useState<any>(null);

	// API URL that works in both development and production Docker environments
	const API_URL = process.env.NEXT_PUBLIC_BACKEND_URL; // In development, backend is on host

	const handleSubmit = async (e: React.FormEvent) => {
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

				// Get encoding method and audio analysis from headers
				const method =
					response.headers.get("X-Encoding-Method") || "Spread Spectrum";
				const capacity = response.headers.get("X-Audio-Capacity");
				const duration = response.headers.get("X-Audio-Duration");

				setEncodingMethod(method);
				setAudioAnalysis({
					capacity,
					duration,
				});

				setProgress(80);
				const blob = await response.blob();
				const audioUrl = URL.createObjectURL(blob);
				setProcessedAudio(audioUrl);
				setEncodedAudioBlob(blob);
				setEncodedFileName(`stego-${audioFile?.name || "audio.wav"}`);
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

	const handleFileChange = (file: File | null) => {
		setAudioFile(file);
		if (file) {
			setError(null);
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
		setAudioAnalysis(null);
	};

	const handleModeChange = (mode: string) => {
		setSelected(mode);
		// Auto-populate decode tab with encoded audio if available
		if (mode === "Decode" && encodedAudioBlob && encodedFileName) {
			const file = new File([encodedAudioBlob], encodedFileName, {
				type: encodedAudioBlob.type || "audio/flac",
			});
			setAudioFile(file);
			setError(null);
		}
	};

	return (
		<div className="container mx-auto py-10 px-4">
			<h1 className="text-3xl font-bold text-center mb-8">
				Audio Steganography
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
					<ModeSelector selected={selected} onModeChange={handleModeChange} />

					<CardHeader>
						<CardTitle>
							{selected === "Encode"
								? " Message Encoding"
								: "Neural Message Decoder"}
						</CardTitle>
						<CardDescription>
							{selected === "Encode"
								? "Our RL agent automatically selects the optimal encoding method for your audio file!"
								: "Extract hidden messages using our trained neural network decoder."}
						</CardDescription>
					</CardHeader>

					<CardContent>
						<form onSubmit={handleSubmit} className="space-y-6">
							<AudioFileInput
								audioFile={audioFile}
								onFileChange={handleFileChange}
								isLoading={isLoading}
								selected={selected}
								encodedAudioBlob={encodedAudioBlob}
								encodedFileName={encodedFileName}
							/>

							{selected === "Encode" && (
								<MessageInput
									message={message}
									onMessageChange={setMessage}
									isLoading={isLoading}
								/>
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
										{selected === "Encode" ? "Encode" : "🔓 Decode"}{" "}
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
											? "Processing..."
											: "Finalizing results..."}
								</p>
							</div>
						)}
					</CardContent>

					{processedAudio && audioFile && selected === "Encode" && (
						<EncodeResults
							audioFile={audioFile}
							processedAudio={processedAudio}
							encodingMethod={encodingMethod}
							encodedFileName={encodedFileName}
							onReset={resetForm}
							audioAnalysis={audioAnalysis}
						/>
					)}

					{decodedMessage && selected === "Decode" && (
						<DecodeResults
							decodedMessage={decodedMessage}
							encodingMethod={encodingMethod}
							onReset={resetForm}
						/>
					)}

					{analysisResults && selected === "Analyze" && (
						<AnalysisResults
							analysisResults={analysisResults}
							onReset={resetForm}
						/>
					)}
				</Card>
			</div>

			<div className="max-w-md mx-auto mt-8 text-center text-sm text-muted-foreground">
				<p>
					🧠 Powered by Reinforcement Learning and Neural Networks for
					intelligent audio steganography. Hidden data is imperceptible to human
					ears while maintaining audio quality.
				</p>
			</div>
		</div>
	);
}
