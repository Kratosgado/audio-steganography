# Audio Steganography Components

This directory contains the separated components for the Audio Steganography application.

## Component Structure

### Main Components

- **`AudioSteganographyApp.tsx`** - The main application component that orchestrates all functionality
- **`ModeSelector.tsx`** - Radio button selector for Encode/Decode/Analyze modes
- **`AudioFileInput.tsx`** - File input component for audio file selection
- **`MessageInput.tsx`** - Textarea component for entering secret messages
- **`EncodeResults.tsx`** - Results display component for encoding operations
- **`DecodeResults.tsx`** - Results display component for decoding operations
- **`AnalysisResults.tsx`** - Comprehensive results display for steganalysis

### Component Dependencies

```
AudioSteganographyApp
├── ModeSelector
├── AudioFileInput
├── MessageInput
├── EncodeResults
├── DecodeResults
└── AnalysisResults
```

## Usage

### Importing Components

```typescript
// Import individual components
import { ModeSelector } from '@/components/ModeSelector';
import { AudioFileInput } from '@/components/AudioFileInput';

// Or import from index
import { ModeSelector, AudioFileInput } from '@/components';
```

### Main App Component

The `AudioSteganographyApp` component manages all state and API interactions:

```typescript
import AudioSteganographyApp from '@/components/AudioSteganographyApp';

export default function Page() {
  return <AudioSteganographyApp />;
}
```

## Component Props

### ModeSelector
- `selected: string` - Currently selected mode
- `onModeChange: (mode: string) => void` - Mode change handler

### AudioFileInput
- `audioFile: File | null` - Selected audio file
- `onFileChange: (file: File | null) => void` - File change handler
- `isLoading: boolean` - Loading state
- `selected: string` - Current mode
- `encodedAudioBlob: Blob | null` - Encoded audio blob for auto-loading
- `encodedFileName: string` - Encoded file name

### MessageInput
- `message: string` - Current message text
- `onMessageChange: (message: string) => void` - Message change handler
- `isLoading: boolean` - Loading state

### EncodeResults
- `audioFile: File` - Original audio file
- `processedAudio: string` - URL of processed audio
- `encodingMethod: string` - Method used for encoding
- `encodedFileName: string` - Name of encoded file
- `onReset: () => void` - Reset handler

### DecodeResults
- `decodedMessage: string` - Decoded message
- `encodingMethod: string` - Method used for decoding
- `onReset: () => void` - Reset handler

### AnalysisResults
- `analysisResults: any` - Analysis results data
- `onReset: () => void` - Reset handler

## Features

- **Modular Design**: Each component has a single responsibility
- **Type Safety**: All components use TypeScript interfaces
- **Reusability**: Components can be easily reused in other parts of the application
- **Maintainability**: Separated concerns make the code easier to maintain and test
- **State Management**: Centralized state management in the main app component 