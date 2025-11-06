import { useState } from 'react';
import { ModelSelection } from './pages/ModelSelection';
import { ExampleGuidance } from './pages/ExampleGuidance';
import { ImageUpload } from './pages/ImageUpload';
import { PredictionResult } from './pages/PredictionResult';

type Step = 'selection' | 'guidance' | 'upload' | 'result';
type ModelType = 'condition' | 'inflation' | null;

export default function App() {
  const [currentStep, setCurrentStep] = useState<Step>('selection');
  const [selectedModel, setSelectedModel] = useState<ModelType>(null);
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);

  const handleModelSelect = (model: 'condition' | 'inflation') => {
    setSelectedModel(model);
    setCurrentStep('guidance');
  };

  const handleProceedToUpload = () => {
    setCurrentStep('upload');
  };

  const handleAnalyze = (imageData: string) => {
    setUploadedImage(imageData);
    setCurrentStep('result');
  };

  const handleReset = () => {
    setCurrentStep('selection');
    setSelectedModel(null);
    setUploadedImage(null);
  };

  const handleUploadAnother = () => {
    setUploadedImage(null);
    setCurrentStep('upload');
  };

  const handleBackFromGuidance = () => {
    setCurrentStep('selection');
    setSelectedModel(null);
  };

  const handleBackFromUpload = () => {
    setCurrentStep('guidance');
  };

  return (
    <div className="min-h-screen">
      {currentStep === 'selection' && (
        <ModelSelection onSelectModel={handleModelSelect} />
      )}
      
      {currentStep === 'guidance' && selectedModel && (
        <ExampleGuidance
          modelType={selectedModel}
          onProceed={handleProceedToUpload}
          onBack={handleBackFromGuidance}
        />
      )}
      
      {currentStep === 'upload' && selectedModel && (
        <ImageUpload
          modelType={selectedModel}
          onAnalyze={handleAnalyze}
          onBack={handleBackFromUpload}
        />
      )}
      
      {currentStep === 'result' && selectedModel && uploadedImage && (
        <PredictionResult
          modelType={selectedModel}
          imageData={uploadedImage}
          onReset={handleReset}
          onUploadAnother={handleUploadAnother}
        />
      )}
    </div>
  );
}
