import React from "react";
import { Button } from "@/components/ui/button";
import { CardFooter } from "@/components/ui/card";
import { FileAudio, Download } from "lucide-react";

interface EncodeResultsProps {
  audioFile: File;
  processedAudio: string;
  encodingMethod: string;
  encodedFileName: string;
  onReset: () => void;
}

export function EncodeResults({
  audioFile,
  processedAudio,
  encodingMethod,
  encodedFileName,
  onReset,
}: EncodeResultsProps) {
  const handleDownload = () => {
    const a = document.createElement("a");
    a.href = processedAudio;
    a.download = encodedFileName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <CardFooter className="flex flex-col gap-4">
      {encodingMethod && (
        <div className="w-full pt-2">
          <div className="bg-green-50 border border-green-200 rounded-lg p-3">
            <p className="text-sm font-medium text-green-800">
              🎯 RL Agent Selected Method: <strong>{encodingMethod}</strong>
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
          <source src={URL.createObjectURL(audioFile)} type={audioFile?.type} />
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
          <Button variant="secondary" className="flex-1" onClick={onReset}>
            Encode Another
          </Button>
          <Button variant="default" className="flex-1" onClick={handleDownload}>
            Download <Download className="ml-2 h-4 w-4" />
          </Button>
        </div>
      </div>
    </CardFooter>
  );
} 