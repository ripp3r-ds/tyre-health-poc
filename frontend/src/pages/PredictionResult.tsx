import { useEffect, useState } from 'react';
import { Card } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Badge } from '../components/ui/badge';
import { Progress } from '../components/ui/progress';
import { CheckCircle2, AlertTriangle, Home, Upload, Ruler, ExternalLink } from 'lucide-react';
import tireSizeImage from '../assets/1663cdcc-86c4-4d00-9f42-998a349a4e0e.png';

interface PredictionResultProps {
  modelType: 'condition' | 'inflation';
  imageData: string;
  onReset: () => void;
  onUploadAnother: () => void;
}

interface PredictionData {
  prediction: string;
  confidence: number;
  isGood: boolean;
}

export function PredictionResult({ modelType, imageData, onReset, onUploadAnother }: PredictionResultProps) {
  const [prediction, setPrediction] = useState<PredictionData | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(true);
  const isCondition = modelType === 'condition';

  useEffect(() => {
    let cancelled = false;

    const apiBase = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
    const endpoint = isCondition ? '/predict/condition' : '/predict/pressure';

    async function dataUrlToSanitizedBlob(dataUrl: string): Promise<Blob> {
      return new Promise<Blob>((resolve, reject) => {
        const img = new Image();
        img.onload = () => {
          try {
            const canvas = document.createElement('canvas');
            canvas.width = img.naturalWidth;
            canvas.height = img.naturalHeight;
            const ctx = canvas.getContext('2d');
            if (!ctx) {
              reject(new Error('Canvas context not available'));
              return;
            }
            ctx.drawImage(img, 0, 0);
            canvas.toBlob((blob) => {
              if (blob) resolve(blob); else reject(new Error('Failed to create blob'));
            }, 'image/jpeg', 0.95);
          } catch (e) {
            reject(e as Error);
          }
        };
        img.onerror = () => reject(new Error('Failed to load image'));
        img.src = dataUrl;
      });
    }

    async function run() {
      try {
        const blob = await dataUrlToSanitizedBlob(imageData);
        const form = new FormData();
        form.append('file', blob, isCondition ? 'condition.jpg' : 'pressure.jpg');

        const res = await fetch(`${apiBase}${endpoint}`, {
          method: 'POST',
          body: form,
        });
        if (!res.ok) throw new Error(`API error ${res.status}`);
        const json = await res.json();

        const label: string = json.prediction;
        const conf01: number = json.confidence; // 0..1 per score.py
        const confidencePct = Math.max(0, Math.min(100, Math.round(conf01 * 1000) / 10));

        const isGoodMapped = isCondition ? (label === 'good') : (label === 'full');
        const displayPrediction = isCondition
          ? (label === 'good' ? 'Good Condition' : 'Worn')
          : (label === 'full' ? 'Properly Inflated' : 'Flat/Under-inflated');

        if (!cancelled) {
          setPrediction({ prediction: displayPrediction, confidence: confidencePct, isGood: isGoodMapped });
          setIsAnalyzing(false);
        }
      } catch (e) {
        if (!cancelled) {
          setPrediction({ prediction: 'Analysis failed', confidence: 0, isGood: false });
          setIsAnalyzing(false);
        }
      }
    }

    run();
    return () => { cancelled = true; };
  }, [isCondition, imageData]);

  const getRecommendation = () => {
    if (!prediction) return '';
    
    if (isCondition) {
      return prediction.isGood
        ? 'Your tire tread appears to be in good condition. Continue regular monitoring and maintenance.'
        : 'Your tire shows signs of wear. Consider having it inspected by a professional and potentially replaced for safety.';
    } else {
      return prediction.isGood
        ? 'Your tire appears to be properly inflated. Maintain regular pressure checks for optimal performance.'
        : 'Your tire may be under-inflated. Check the pressure with a gauge and inflate to the recommended PSI listed on your vehicle door jamb.';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-4 py-12 animate-in fade-in duration-500">
      <div className="max-w-4xl mx-auto space-y-8">
        <div className="text-center space-y-4 animate-in slide-in-from-bottom-4 duration-700">
          <Badge className={`${isCondition ? 'bg-blue-600' : 'bg-emerald-600'} text-white px-4 py-1.5 shadow-lg`}>
            {isCondition ? 'Tire Condition Model' : 'Tire Inflation Model'}
          </Badge>
          <h1 className="text-white">Analysis Results</h1>
        </div>

        {isAnalyzing ? (
          <Card className="bg-slate-800/50 border-slate-700 backdrop-blur-sm p-12 animate-in zoom-in-95 duration-500">
            <div className="space-y-6 text-center">
              <div className="w-20 h-20 mx-auto relative">
                <div className={`w-full h-full border-4 border-slate-600 ${isCondition ? 'border-t-blue-500' : 'border-t-emerald-500'} rounded-full animate-spin`}></div>
                <div className={`absolute inset-0 ${isCondition ? 'bg-blue-500' : 'bg-emerald-500'} rounded-full opacity-20 animate-ping`}></div>
              </div>
              <div className="space-y-2 animate-in fade-in duration-1000" style={{ animationDelay: '200ms' }}>
                <p className="text-white">Analyzing Your Tire...</p>
                <p className="text-slate-400">Processing image through ML model</p>
                <div className="flex justify-center gap-1 pt-2">
                  <div className={`w-2 h-2 rounded-full ${isCondition ? 'bg-blue-500' : 'bg-emerald-500'} animate-bounce`} style={{ animationDelay: '0ms' }}></div>
                  <div className={`w-2 h-2 rounded-full ${isCondition ? 'bg-blue-500' : 'bg-emerald-500'} animate-bounce`} style={{ animationDelay: '150ms' }}></div>
                  <div className={`w-2 h-2 rounded-full ${isCondition ? 'bg-blue-500' : 'bg-emerald-500'} animate-bounce`} style={{ animationDelay: '300ms' }}></div>
                </div>
              </div>
            </div>
          </Card>
        ) : prediction ? (
          <div className="grid md:grid-cols-5 gap-6 animate-in fade-in duration-700">
            <div className="md:col-span-2 animate-in slide-in-from-left duration-700">
              <Card className="bg-slate-800/50 border-slate-700 backdrop-blur-sm p-6 space-y-4 hover:bg-slate-800/60 hover:scale-[1.02] transition-all">
                <h3 className="text-white">Uploaded Image</h3>
                <div className="rounded-lg overflow-hidden bg-slate-900/50 ring-1 ring-slate-600/50 hover:ring-slate-500/50 transition-all">
                  <img
                    src={imageData}
                    alt="Analyzed tire"
                    className="w-full h-auto object-contain"
                  />
                </div>
              </Card>
            </div>

            <div className="md:col-span-3 space-y-6 animate-in slide-in-from-right duration-700">
              <Card className={`border-2 ${prediction.isGood ? 'bg-emerald-500/10 border-emerald-500/50 shadow-2xl shadow-emerald-500/20' : 'bg-amber-500/10 border-amber-500/50 shadow-2xl shadow-amber-500/20'} backdrop-blur-sm p-8 hover:scale-[1.02] transition-all duration-300`}>
                <div className="space-y-6">
                  <div className="flex items-start gap-4 animate-in zoom-in-95 duration-500">
                    {prediction.isGood ? (
                      <div className="w-12 h-12 bg-emerald-500/20 rounded-full flex items-center justify-center flex-shrink-0 animate-in zoom-in-50 duration-700">
                        <CheckCircle2 className="w-6 h-6 text-emerald-400" />
                      </div>
                    ) : (
                      <div className="w-12 h-12 bg-amber-500/20 rounded-full flex items-center justify-center flex-shrink-0 animate-in zoom-in-50 duration-700">
                        <AlertTriangle className="w-6 h-6 text-amber-400" />
                      </div>
                    )}
                    <div className="flex-1 space-y-2">
                      <h2 className="text-white">{prediction.prediction}</h2>
                      <p className={`${prediction.isGood ? 'text-emerald-400' : 'text-amber-400'}`}>
                        Model Confidence: {prediction.confidence.toFixed(1)}%
                      </p>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-slate-400">Confidence Level</span>
                      <span className="text-slate-300">{prediction.confidence.toFixed(1)}%</span>
                    </div>
                    <Progress 
                      value={prediction.confidence} 
                      className="h-3 bg-slate-700"
                    />
                  </div>
                </div>
              </Card>

              <Card className="bg-slate-800/50 border-slate-700 backdrop-blur-sm p-6 space-y-4 hover:bg-slate-800/60 hover:scale-[1.02] transition-all animate-in slide-in-from-bottom-4 duration-700" style={{ animationDelay: '100ms' }}>
                <h3 className="text-white">Recommendation</h3>
                <p className="text-slate-300 leading-relaxed">
                  {getRecommendation()}
                </p>
              </Card>

              <Card className="bg-slate-800/50 border-slate-700 backdrop-blur-sm p-6 space-y-4 hover:bg-slate-800/60 hover:scale-[1.02] transition-all animate-in slide-in-from-bottom-4 duration-700" style={{ animationDelay: '200ms' }}>
                <h3 className="text-white">Model Information</h3>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="text-slate-400">Model Type</p>
                    <p className="text-slate-200">Binary Classification</p>
                  </div>
                  <div>
                    <p className="text-slate-400">Classes</p>
                    <p className="text-slate-200">
                      {isCondition ? 'Good / Worn' : 'Full / Flat'}
                    </p>
                  </div>
                  <div>
                    <p className="text-slate-400">Framework</p>
                    <p className="text-slate-200">TensorFlow / PyTorch</p>
                  </div>
                  <div>
                    <p className="text-slate-400">Processing</p>
                    <p className="text-slate-200">Client-side</p>
                  </div>
                </div>
              </Card>

              {/* Condition Model: Tire Size Guide */}
              {isCondition && (
                <Card className="bg-blue-500/10 border-blue-500/30 backdrop-blur-sm p-6 space-y-4 hover:bg-blue-500/15 hover:scale-[1.02] transition-all animate-in slide-in-from-bottom-4 duration-700" style={{ animationDelay: '300ms' }}>
                  <div className="flex items-start gap-3">
                    <Ruler className="w-5 h-5 text-blue-400 flex-shrink-0 mt-1" />
                    <div className="space-y-3 flex-1">
                      <h3 className="text-white">Want to know your tire size for looking up replacements?</h3>
                      <p className="text-slate-300 text-sm">
                        Understanding your tire size is crucial when shopping for replacements. Here's how to read it:
                      </p>
                      <div className="bg-slate-900/50 rounded-lg p-4">
                        <img 
                          src={tireSizeImage} 
                          alt="Tire size guide showing width, aspect ratio, construction type, and rim diameter"
                          className="w-full h-auto"
                        />
                      </div>
                      <p className="text-slate-400 text-sm">
                        You can find these numbers on the sidewall of your tire. For example, <span className="text-white">195/55 R16</span> means 195mm width, 55% aspect ratio, Radial construction, and 16-inch rim diameter.
                      </p>
                    </div>
                  </div>
                </Card>
              )}

              {/* Inflation Model: PSI Finder */}
              {!isCondition && (
                <Card className="bg-emerald-500/10 border-emerald-500/30 backdrop-blur-sm p-6 space-y-4 hover:bg-emerald-500/15 hover:scale-[1.02] transition-all animate-in slide-in-from-bottom-4 duration-700" style={{ animationDelay: '300ms' }}>
                  <div className="flex items-start gap-3">
                    <AlertTriangle className="w-5 h-5 text-emerald-400 flex-shrink-0 mt-1" />
                    <div className="space-y-3 flex-1">
                      <h3 className="text-white">Need to know the right PSI for your tire?</h3>
                      <p className="text-slate-300 text-sm">
                        Every vehicle has a recommended tire pressure. Don't guess—find the exact PSI for your specific vehicle.
                      </p>
                      <Button
                        onClick={() => window.open('https://chatgpt.com/?q=What+is+the+recommended+tire+pressure+PSI+for+my+vehicle', '_blank')}
                        className="w-full bg-emerald-600 hover:bg-emerald-700 shadow-lg shadow-emerald-600/30 hover:scale-105 transition-all"
                      >
                        Find My Recommended PSI
                        <ExternalLink className="w-4 h-4 ml-2" />
                      </Button>
                      <p className="text-slate-400 text-sm">
                        Tip: Check your vehicle's door jamb sticker or owner's manual for the manufacturer's recommended PSI.
                      </p>
                    </div>
                  </div>
                </Card>
              )}

              <div className="flex gap-4 animate-in slide-in-from-bottom-4 duration-700" style={{ animationDelay: '400ms' }}>
                <Button
                  onClick={onUploadAnother}
                  variant="outline"
                  className="flex-1 border-slate-600 text-slate-300 hover:bg-slate-800 hover:scale-105 transition-all"
                >
                  <Upload className="w-4 h-4 mr-2" />
                  Analyze Another Image
                </Button>
                <Button
                  onClick={onReset}
                  className={`flex-1 ${isCondition ? 'bg-blue-600 hover:bg-blue-700 shadow-lg shadow-blue-600/30' : 'bg-emerald-600 hover:bg-emerald-700 shadow-lg shadow-emerald-600/30'} hover:scale-105 transition-all`}
                >
                  <Home className="w-4 h-4 mr-2" />
                  Back to Home
                </Button>
              </div>
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}
