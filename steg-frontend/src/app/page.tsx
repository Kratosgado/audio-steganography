// import Image from "next/image";

// export default function Home() {
//   return (
//     <div className="grid grid-rows-[20px_1fr_20px] items-center justify-items-center min-h-screen p-8 pb-20 gap-16 sm:p-20 font-[family-name:var(--font-geist-sans)]">
//       <main className="flex flex-col gap-8 row-start-2 items-center sm:items-start">
//         <Image
//           className="dark:invert"
//           src="/next.svg"
//           alt="Next.js logo"
//           width={180}
//           height={38}
//           priority
//         />
//         <ol className="list-inside list-decimal text-sm text-center sm:text-left font-[family-name:var(--font-geist-mono)]">
//           <li className="mb-2">
//             Get started by editing{" "}
//             <code className="bg-black/[.05] dark:bg-white/[.06] px-1 py-0.5 rounded font-semibold">
//               src/app/page.tsx
//             </code>
//             .
//           </li>
//           <li>Save and see your changes instantly.</li>
//         </ol>

//         <div className="flex gap-4 items-center flex-col sm:flex-row">
//           <a
//             className="rounded-full border border-solid border-transparent transition-colors flex items-center justify-center bg-foreground text-background gap-2 hover:bg-[#383838] dark:hover:bg-[#ccc] text-sm sm:text-base h-10 sm:h-12 px-4 sm:px-5"
//             href="https://vercel.com/new?utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
//             target="_blank"
//             rel="noopener noreferrer"
//           >
//             <Image
//               className="dark:invert"
//               src="/vercel.svg"
//               alt="Vercel logomark"
//               width={20}
//               height={20}
//             />
//             Deploy now
//           </a>
//           <a
//             className="rounded-full border border-solid border-black/[.08] dark:border-white/[.145] transition-colors flex items-center justify-center hover:bg-[#f2f2f2] dark:hover:bg-[#1a1a1a] hover:border-transparent text-sm sm:text-base h-10 sm:h-12 px-4 sm:px-5 sm:min-w-44"
//             href="https://nextjs.org/docs?utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
//             target="_blank"
//             rel="noopener noreferrer"
//           >
//             Read our docs
//           </a>
//         </div>
//       </main>
//       <footer className="row-start-3 flex gap-6 flex-wrap items-center justify-center">
//         <a
//           className="flex items-center gap-2 hover:underline hover:underline-offset-4"
//           href="https://nextjs.org/learn?utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
//           target="_blank"
//           rel="noopener noreferrer"
//         >
//           <Image
//             aria-hidden
//             src="/file.svg"
//             alt="File icon"
//             width={16}
//             height={16}
//           />
//           Learn
//         </a>
//         <a
//           className="flex items-center gap-2 hover:underline hover:underline-offset-4"
//           href="https://vercel.com/templates?framework=next.js&utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
//           target="_blank"
//           rel="noopener noreferrer"
//         >
//           <Image
//             aria-hidden
//             src="/window.svg"
//             alt="Window icon"
//             width={16}
//             height={16}
//           />
//           Examples
//         </a>
//         <a
//           className="flex items-center gap-2 hover:underline hover:underline-offset-4"
//           href="https://nextjs.org?utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
//           target="_blank"
//           rel="noopener noreferrer"
//         >
//           <Image
//             aria-hidden
//             src="/globe.svg"
//             alt="Globe icon"
//             width={16}
//             height={16}
//           />
//           Go to nextjs.org →
//         </a>
//       </footer>
//     </div>
//   );
// }
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
    } catch (err) {
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

