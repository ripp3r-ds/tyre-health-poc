import { Card } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Gauge, Eye, AlertTriangle, TrendingDown, DollarSign } from 'lucide-react';
import { ImageWithFallback } from './ImageWithFallback';

interface ModelSelectionProps {
  onSelectModel: (model: 'condition' | 'inflation') => void;
}

export function ModelSelection({ onSelectModel }: ModelSelectionProps) {
  const stats = [
    {
      icon: AlertTriangle,
      value: '11,000+',
      label: 'Tire-related crashes annually',
      color: 'text-red-400'
    },
    {
      icon: TrendingDown,
      value: '25%',
      label: 'Reduced fuel efficiency from under-inflation',
      color: 'text-amber-400'
    },
    {
      icon: DollarSign,
      value: '$600+',
      label: 'Average cost of tire replacement set',
      color: 'text-emerald-400'
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 animate-in fade-in duration-500">
      {/* Hero Section with Background */}
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 opacity-30">
          <ImageWithFallback
            src="https://images.unsplash.com/photo-1647698848461-8b6ae7df53b0?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHx0aXJlJTIwZ3JvdW5kJTIwdHJlYWQlMjBsb3clMjBhbmdsZXxlbnwxfHx8fDE3NjIxNzY5NTJ8MA&ixlib=rb-4.1.0&q=80&w=1080"
            alt="Tire tread background"
            className="w-full h-full object-cover"
          />
          <div className="absolute inset-0 bg-gradient-to-b from-slate-900/60 via-slate-900/70 to-slate-900"></div>
        </div>
        
        <div className="relative max-w-6xl mx-auto px-4 py-16 md:py-24">
          <div className="text-center space-y-6 mb-12 animate-in slide-in-from-bottom-4 duration-700">
            <h1 className="text-white">TireCheck AI</h1>
            <p className="text-slate-300 max-w-3xl mx-auto text-lg">
              Your tires are the only contact between your vehicle and the road. Regular inspection can prevent accidents, improve fuel efficiency, and save you money.
            </p>
          </div>

          {/* Stats Grid */}
          <div className="grid md:grid-cols-3 gap-6 mb-16">
            {stats.map((stat, index) => (
              <Card 
                key={index} 
                className="bg-slate-800/60 border-slate-700 backdrop-blur-sm p-6 hover:bg-slate-800/80 transition-all duration-300 hover:scale-105 hover:shadow-lg hover:shadow-slate-900/50 animate-in slide-in-from-bottom-4 duration-700"
                style={{ animationDelay: `${index * 100}ms` }}
              >
                <div className="flex items-start gap-4">
                  <div className={`w-12 h-12 rounded-xl bg-slate-900/50 flex items-center justify-center flex-shrink-0`}>
                    <stat.icon className={`w-6 h-6 ${stat.color}`} />
                  </div>
                  <div className="space-y-1">
                    <p className="text-white text-2xl">{stat.value}</p>
                    <p className="text-slate-400 text-sm">{stat.label}</p>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        </div>
      </div>

      {/* Model Selection Section */}
      <div className="max-w-4xl mx-auto px-4 pb-16">
        <div className="text-center space-y-4 mb-10 animate-in slide-in-from-bottom-4 duration-700" style={{ animationDelay: '400ms' }}>
          <h2 className="text-white">Choose Your Inspection Type</h2>
          <p className="text-slate-300 max-w-2xl mx-auto">
            Select which aspect of your tire you'd like to analyze using our ML-powered classification models.
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-6 mb-8">
          <Card 
            className="bg-slate-800/60 border-slate-700 backdrop-blur-sm hover:bg-slate-800/80 hover:border-blue-500/50 hover:shadow-2xl hover:shadow-blue-500/10 transition-all duration-300 cursor-pointer group hover:scale-105 animate-in slide-in-from-bottom-4 duration-700"
            style={{ animationDelay: '500ms' }}
            onClick={() => onSelectModel('condition')}
          >
            <div className="p-8 space-y-4">
              <div className="w-16 h-16 bg-blue-500/10 rounded-2xl flex items-center justify-center group-hover:bg-blue-500/20 group-hover:scale-110 transition-all duration-300">
                <Eye className="w-8 h-8 text-blue-400 group-hover:text-blue-300 transition-colors" />
              </div>
              <div className="space-y-2">
                <h2 className="text-white group-hover:text-blue-100 transition-colors">Tire Condition</h2>
                <p className="text-slate-400 group-hover:text-slate-300 transition-colors">
                  Analyze tread wear patterns to determine if your tire is in good condition or needs replacement.
                </p>
              </div>
              <div className="pt-4">
                <Button 
                  className="w-full bg-blue-600 hover:bg-blue-700 shadow-lg shadow-blue-600/30 group-hover:shadow-blue-600/50 transition-all"
                  onClick={(e) => {
                    e.stopPropagation();
                    onSelectModel('condition');
                  }}
                >
                  Check Condition
                </Button>
              </div>
            </div>
          </Card>

          <Card 
            className="bg-slate-800/60 border-slate-700 backdrop-blur-sm hover:bg-slate-800/80 hover:border-emerald-500/50 hover:shadow-2xl hover:shadow-emerald-500/10 transition-all duration-300 cursor-pointer group hover:scale-105 animate-in slide-in-from-bottom-4 duration-700"
            style={{ animationDelay: '600ms' }}
            onClick={() => onSelectModel('inflation')}
          >
            <div className="p-8 space-y-4">
              <div className="w-16 h-16 bg-emerald-500/10 rounded-2xl flex items-center justify-center group-hover:bg-emerald-500/20 group-hover:scale-110 transition-all duration-300">
                <Gauge className="w-8 h-8 text-emerald-400 group-hover:text-emerald-300 transition-colors" />
              </div>
              <div className="space-y-2">
                <h2 className="text-white group-hover:text-emerald-100 transition-colors">Tire Inflation</h2>
                <p className="text-slate-400 group-hover:text-slate-300 transition-colors">
                  Detect whether your tire is properly inflated or requires air pressure adjustment.
                </p>
              </div>
              <div className="pt-4">
                <Button 
                  className="w-full bg-emerald-600 hover:bg-emerald-700 shadow-lg shadow-emerald-600/30 group-hover:shadow-emerald-600/50 transition-all"
                  onClick={(e) => {
                    e.stopPropagation();
                    onSelectModel('inflation');
                  }}
                >
                  Check Inflation
                </Button>
              </div>
            </div>
          </Card>
        </div>

        <div className="text-center text-slate-500 text-sm pt-4 animate-in fade-in duration-1000" style={{ animationDelay: '700ms' }}>
          Portfolio ML Classification Project • Binary Classification Models
        </div>
      </div>
    </div>
  );
}
