import { useState, useRef } from 'react';
import { Card } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Badge } from '../components/ui/badge';
import { Upload, ArrowLeft, Image as ImageIcon, X } from 'lucide-react';

interface ImageUploadProps {
  modelType: 'condition' | 'inflation';
  onAnalyze: (imageData: string) => void;
  onBack: () => void;
}

export function ImageUpload({ modelType, onAnalyze, onBack }: ImageUploadProps) {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const isCondition = modelType === 'condition';

  const handleFileSelect = (file: File) => {
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const result = e.target?.result as string;
        setSelectedImage(result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFileSelect(file);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFileSelect(file);
  };

  const clearImage = () => {
    setSelectedImage(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-4 py-12 animate-in fade-in duration-500">
      <div className="max-w-3xl mx-auto space-y-8">
        <div className="flex items-center justify-between animate-in slide-in-from-left duration-500">
          <Button
            variant="ghost"
            onClick={onBack}
            className="text-slate-400 hover:text-white hover:bg-slate-800/50 border border-slate-700 hover:border-slate-600 transition-all"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back
          </Button>
          <div className="text-slate-500 text-sm">Step 3 of 3</div>
        </div>

        <div className="text-center space-y-4 animate-in slide-in-from-bottom-4 duration-700">
          <Badge className={`${isCondition ? 'bg-blue-600' : 'bg-emerald-600'} text-white px-4 py-1.5 shadow-lg`}>
            {isCondition ? 'Tire Condition Model' : 'Tire Inflation Model'}
          </Badge>
          <h1 className="text-white">Upload Your Tire Photo</h1>
          <p className="text-slate-300 max-w-2xl mx-auto">
            Upload a clear photo of your tire for analysis. Our ML model will process it and provide results instantly.
          </p>
        </div>

        <Card className="bg-slate-800/50 border-slate-700 backdrop-blur-sm p-8 hover:bg-slate-800/60 transition-colors animate-in slide-in-from-bottom-4 duration-700">
          {!selectedImage ? (
            <div
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              className={`border-2 border-dashed rounded-xl p-12 text-center space-y-4 transition-all cursor-pointer ${
                isDragging
                  ? isCondition 
                    ? 'border-blue-500 bg-blue-500/10 scale-105' 
                    : 'border-emerald-500 bg-emerald-500/10 scale-105'
                  : 'border-slate-600 hover:border-slate-500 hover:bg-slate-700/20'
              }`}
              onClick={() => fileInputRef.current?.click()}
            >
              <div className={`w-20 h-20 mx-auto rounded-full flex items-center justify-center transition-all ${
                isDragging 
                  ? isCondition 
                    ? 'bg-blue-500/20 scale-110' 
                    : 'bg-emerald-500/20 scale-110'
                  : 'bg-slate-700/50'
              }`}>
                <Upload className={`w-10 h-10 transition-all ${
                  isDragging 
                    ? isCondition 
                      ? 'text-blue-400 animate-pulse' 
                      : 'text-emerald-400 animate-pulse'
                    : 'text-slate-400'
                }`} />
              </div>
              <div className="space-y-2">
                <p className="text-white">
                  Drag and drop your image here
                </p>
                <p className="text-slate-400">
                  or click to browse files
                </p>
              </div>
              <p className="text-slate-500 text-sm">
                Supports: JPG, PNG, WebP (Max 10MB)
              </p>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileInput}
                className="hidden"
              />
            </div>
          ) : (
            <div className="space-y-6 animate-in zoom-in-95 duration-500">
              <div className="relative rounded-xl overflow-hidden bg-slate-900/50 ring-1 ring-slate-600/50 hover:ring-slate-500/50 transition-all">
                <img
                  src={selectedImage}
                  alt="Selected tire"
                  className="w-full h-auto max-h-96 object-contain mx-auto"
                />
                <Button
                  variant="ghost"
                  onClick={clearImage}
                  className="absolute top-4 right-4 bg-slate-900/90 hover:bg-slate-900 text-white backdrop-blur-sm hover:scale-105 transition-transform"
                >
                  <X className="w-4 h-4 mr-2" />
                  Remove
                </Button>
              </div>
              
              <div className="flex gap-4">
                <Button
                  variant="outline"
                  onClick={() => fileInputRef.current?.click()}
                  className="flex-1 border-slate-600 text-slate-300 hover:bg-slate-800 hover:scale-105 transition-all"
                >
                  <ImageIcon className="w-4 h-4 mr-2" />
                  Choose Different Image
                </Button>
                <Button
                  onClick={() => selectedImage && onAnalyze(selectedImage)}
                  className={`flex-1 ${isCondition ? 'bg-blue-600 hover:bg-blue-700 shadow-lg shadow-blue-600/30' : 'bg-emerald-600 hover:bg-emerald-700 shadow-lg shadow-emerald-600/30'} hover:scale-105 transition-all`}
                >
                  Analyze Image
                </Button>
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileInput}
                className="hidden"
              />
            </div>
          )}
        </Card>

        <Card className="bg-blue-500/10 border-blue-500/30 backdrop-blur-sm p-6 animate-in slide-in-from-bottom-4 duration-700" style={{ animationDelay: '200ms' }}>
          <div className="flex gap-4">
            <div className="text-blue-400 flex-shrink-0">
              <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <div className="space-y-1">
              <p className="text-white">Privacy Notice</p>
              <p className="text-slate-400 text-sm">
                Your image is sent to your local API at <span className="text-white">http://localhost:8000</span> for analysis. Metadata (EXIF) is stripped client‑side before upload.
              </p>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}
