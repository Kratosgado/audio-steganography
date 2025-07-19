import React from "react";
import { Button } from "@/components/ui/button";
import { CardFooter } from "@/components/ui/card";

interface DecodeResultsProps {
  decodedMessage: string;
  encodingMethod: string;
  onReset: () => void;
}

export function DecodeResults({
  decodedMessage,
  encodingMethod,
  onReset,
}: DecodeResultsProps) {
  const handleCopyMessage = () => {
    navigator.clipboard.writeText(decodedMessage);
  };

  return (
    <CardFooter className="flex flex-col gap-4">
      {encodingMethod && (
        <div className="w-full pt-2">
          <div className="bg-green-50 border border-green-200 rounded-lg p-3">
            <p className="text-sm font-medium text-green-800">
              🤖 RL Agent Decoding Method: <strong>{encodingMethod}</strong>
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
          <p className="font-mono text-sm break-words">{decodedMessage}</p>
        </div>
        <div className="grid grid-cols-2 gap-4 text-sm mb-4">
          <div>
            <p className="text-gray-600">Message Length</p>
            <p className="font-medium">{decodedMessage.length} characters</p>
          </div>
          <div>
            <p className="text-gray-600">Decoding Success</p>
            <p className="font-medium text-green-600">✅ Complete</p>
          </div>
        </div>
        <div className="flex gap-2 mt-4">
          <Button variant="secondary" className="flex-1" onClick={onReset}>
            Decode Another
          </Button>
          <Button variant="outline" className="flex-1" onClick={handleCopyMessage}>
            Copy Message
          </Button>
        </div>
      </div>
    </CardFooter>
  );
} 