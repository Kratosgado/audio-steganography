import React from "react";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";

interface MessageInputProps {
  message: string;
  onMessageChange: (message: string) => void;
  isLoading: boolean;
}

export function MessageInput({ message, onMessageChange, isLoading }: MessageInputProps) {
  return (
    <div className="space-y-2">
      <Label htmlFor="message">Secret Message</Label>
      <Textarea
        id="message"
        placeholder="Enter the message you want to hide..."
        value={message}
        onChange={(e) => onMessageChange(e.target.value)}
        disabled={isLoading}
        className="min-h-[100px]"
      />
      <p className="text-xs text-gray-500">
        💡 Our RL agent will automatically choose the best encoding method for your audio
      </p>
    </div>
  );
} 