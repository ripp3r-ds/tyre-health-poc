import { Card } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Badge } from '../components/ui/badge';
import { ImageWithFallback } from './ImageWithFallback';
import { CheckCircle2, XCircle, ArrowRight, ArrowLeft } from 'lucide-react';

interface ExampleGuidanceProps {
  modelType: 'condition' | 'inflation';
  onProceed: () => void;
  onBack: () => void;
}

export function ExampleGuidance({ modelType, onProceed, onBack }: ExampleGuidanceProps) {
  const isCondition = modelType === 'condition';
  
  const goodExamples = isCondition ? [
    {
      src: 'https://images.unsplash.com/photo-1682409408723-8eab5a06477a?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxuZXclMjB0aXJlJTIwdHJlYWR8ZW58MXx8fHwxNzYyMTc0NjY3fDA&ixlib=rb-4.1.0&q=80&w=1080',
      label: 'Clear tread view'
    },
    {
      src: 'https://images.unsplash.com/photo-1694065628369-eaa8347e4337?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxjYXIlMjB0aXJlJTIwY2xvc2V8ZW58MXx8fHwxNzYyMTc0NjY2fDA&ixlib=rb-4.1.0&q=80&w=1080',
      label: 'Close-up shot'
    }
  ] : [
    {
      src: 'https://plus.unsplash.com/premium_photo-1694670121843-79b433c4ed59?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&q=80&w=774',
      label: 'Full tire visible'
    },
    {
      src: 'https://images.unsplash.com/photo-1706743304794-9e96df760a6b?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&q=80&w=1548',
      label: 'Side profile view'
    }
  ];

  const tips = isCondition ? [
    'Ensure the tire tread is clearly visible and in focus',
    'Take the photo in good lighting conditions',
    'Capture the tread pattern from directly above',
    'Avoid shadows or reflections on the tire surface'
  ] : [
    'Photograph the entire tire from the side',
    'Include the wheel rim for better context',
    'Ensure the tire is visible against a clear background',
    'Capture the profile to show any deflation'
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-4 py-10 animate-in fade-in duration-500">
      <div className="max-w-5xl mx-auto space-y-6">
        <div className="flex items-center justify-between animate-in slide-in-from-left duration-500">
          <Button
            variant="ghost"
            onClick={onBack}
            className="text-slate-400 hover:text-white hover:bg-slate-800/50 border border-slate-700 hover:border-slate-600 transition-all"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back
          </Button>
          <div className="text-slate-500 text-sm">Step 2 of 3</div>
        </div>

        <div className="text-center space-y-4 animate-in slide-in-from-bottom-4 duration-700">
          <Badge className={`${isCondition ? 'bg-blue-600' : 'bg-emerald-600'} text-white px-4 py-1.5 shadow-lg`}>
            {isCondition ? 'Tire Condition Model' : 'Tire Inflation Model'}
          </Badge>
          <h1 className="text-white text-2xl md:text-4xl font-semibold">Photo Guidelines</h1>
          <p className="text-slate-200 max-w-2xl mx-auto">
            For optimal results, follow these guidelines when capturing your tire photo. The model performs best with clear, well-lit images.
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          <Card className="bg-slate-800/50 border-slate-700 backdrop-blur-sm p-6 space-y-4 hover:bg-slate-800/60 transition-all duration-300 hover:scale-[1.02] animate-in slide-in-from-left duration-700">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 bg-emerald-500/10 rounded-lg flex items-center justify-center">
                <CheckCircle2 className="w-5 h-5 text-emerald-400" />
              </div>
              <h3 className="text-white">Good Examples</h3>
            </div>
            <div className="grid grid-cols-2 gap-4">
              {goodExamples.map((example, index) => (
                <div key={index} className="space-y-2 group">
                  <div className="aspect-square rounded-lg overflow-hidden bg-slate-700/50 ring-1 ring-slate-600/50 group-hover:ring-emerald-500/30 transition-all duration-300">
                    <ImageWithFallback
                      src={example.src}
                      alt={example.label}
                      className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-300"
                    />
                  </div>
                  <p className="text-slate-400 text-sm text-center group-hover:text-slate-300 transition-colors">{example.label}</p>
                </div>
              ))}
            </div>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700 backdrop-blur-sm p-6 space-y-4 hover:bg-slate-800/60 transition-all duration-300 hover:scale-[1.02] animate-in slide-in-from-right duration-700">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 bg-red-500/10 rounded-lg flex items-center justify-center">
                <XCircle className="w-5 h-5 text-red-400" />
              </div>
              <h3 className="text-white">Avoid These</h3>
            </div>
            <ul className="space-y-3">
              <li className="flex gap-3 text-slate-400 hover:text-slate-300 transition-colors">
                <span className="text-red-400">•</span>
                <span>Blurry or out-of-focus images</span>
              </li>
              <li className="flex gap-3 text-slate-400 hover:text-slate-300 transition-colors">
                <span className="text-red-400">•</span>
                <span>Poor lighting or heavy shadows</span>
              </li>
              <li className="flex gap-3 text-slate-400 hover:text-slate-300 transition-colors">
                <span className="text-red-400">•</span>
                <span>Distant shots where tire details aren't visible</span>
              </li>
              <li className="flex gap-3 text-slate-400 hover:text-slate-300 transition-colors">
                <span className="text-red-400">•</span>
                <span>Obstructed views or partial tire coverage</span>
              </li>
            </ul>
          </Card>
        </div>

        <Card className="bg-slate-800/50 border-slate-700 backdrop-blur-sm p-6 hover:bg-slate-800/60 transition-colors animate-in slide-in-from-bottom-4 duration-700" style={{ animationDelay: '200ms' }}>
          <div className="space-y-4">
            <h3 className="text-white">Pro Tips</h3>
            <div className="grid md:grid-cols-2 gap-3">
              {tips.map((tip, index) => (
                <div key={index} className="flex gap-3 text-slate-400 hover:text-slate-300 transition-colors">
                  <CheckCircle2 className="w-5 h-5 text-emerald-400 flex-shrink-0 mt-0.5" />
                  <span>{tip}</span>
                </div>
              ))}
            </div>
          </div>
        </Card>

        <div className="flex justify-center pt-4 animate-in slide-in-from-bottom-4 duration-700" style={{ animationDelay: '400ms' }}>
          <Button
            onClick={onProceed}
            className={`${isCondition ? 'bg-blue-600 hover:bg-blue-700 shadow-lg shadow-blue-600/30' : 'bg-emerald-600 hover:bg-emerald-700 shadow-lg shadow-emerald-600/30'} px-8 hover:scale-105 transition-all duration-300`}
          >
            I Understand, Proceed to Upload
            <ArrowRight className="w-4 h-4 ml-2" />
          </Button>
        </div>
      </div>
    </div>
  );
}
