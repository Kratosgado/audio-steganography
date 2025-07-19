import React, { ChangeEvent } from "react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

interface AudioFileInputProps {
  audioFile: File | null;
  onFileChange: (file: File | null) => void;
  isLoading: boolean;
  selected: string;
  encodedAudioBlob: Blob | null;
  encodedFileName: string;
}

export function AudioFileInput({
  audioFile,
  onFileChange,
  isLoading,
  selected,
  encodedAudioBlob,
  encodedFileName,
}: AudioFileInputProps) {
  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (file.type.startsWith("audio/")) {
        onFileChange(file);
      } else {
        onFileChange(null);
      }
    }
  };

  return (
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
            Selected: {audioFile.name} ({(audioFile.size / 1024).toFixed(2)} KB)
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
  );
} 