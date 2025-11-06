import { Card } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Gauge, Eye, AlertTriangle, TrendingDown, DollarSign, Shield, Droplets, Clock, Github } from 'lucide-react';
import { ImageWithFallback } from './ImageWithFallback';
import logo from '../assets/logo.svg';

interface ModelSelectionProps {
  onSelectModel: (model: 'condition' | 'inflation') => void;
}

export function ModelSelection({ onSelectModel }: ModelSelectionProps) {
  const stats = [
    { icon: AlertTriangle, value: '11,000+', label: 'Tire-related crashes annually', color: 'text-red-400' },
    { icon: TrendingDown, value: '25%',    label: 'Reduced fuel efficiency from under-inflation', color: 'text-amber-400' },
    { icon: DollarSign,   value: '$600+',  label: 'Average cost of tire replacement set', color: 'text-emerald-400' },
    { icon: Shield,       value: '40%+',   label: 'Of breakdowns linked to tire issues', color: 'text-blue-400' },
    { icon: Droplets,     value: '3x',     label: 'Longer braking distance when under-inflated', color: 'text-cyan-400' },
    { icon: Clock,        value: 'Monthly',label: 'Recommended cadence for tire pressure checks', color: 'text-violet-400' },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 animate-in fade-in duration-500">
      {/* Header */}
      <header className="relative z-10">
        <div className="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between animate-in slide-in-from-top-2 duration-500">
          <div className="flex items-center gap-2">
            <img
              src={logo}
              alt="TireCheck AI logo"
              className="w-6 h-6 transition-transform hover:scale-110 filter brightness-0 invert"

            />
            <span className="text-white/90 font-semibold tracking-tight">TireCheck AI</span>
          </div>
          <a
            href="https://github.com/ripp3r-ds/tyre-health-poc"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 text-sm text-slate-300 hover:text-white transition-colors"
            aria-label="View project on GitHub"
          >
            <Github className="size-4" />
            GitHub
          </a>
        </div>
      </header>

      {/* Hero with background */}
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 opacity-30">
          <ImageWithFallback
            src="https://images.unsplash.com/photo-1647698848461-8b6ae7df53b0?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHx0aXJlJTIwZ3JvdW5kJTIwdHJlYWQlMjBsb3clMjBhbmdsZXxlbnwxfHx8fDE3NjIxNzY5NTJ8MA&ixlib=rb-4.1.0&q=80&w=1080"
            alt="Tire tread background"
            className="w-full h-full object-cover"
          />
          <div className="absolute inset-0 bg-gradient-to-b from-slate-900/50 via-slate-900/70 to-slate-900" />
        </div>

        <div className="relative max-w-6xl mx-auto px-4 py-4 md:py-8">
          <div className="text-center space-y-2 mb-4 animate-in slide-in-from-bottom-4 duration-700">
            <h1 className="text-white text-3xl md:text-5xl font-semibold tracking-tight">TireCheck AI</h1>
            <p className="text-slate-200 max-w-3xl mx-auto text-base md:text-lg">
              Quick, AI-assisted checks to spot common tire issues before they get expensive.
              Built for academic demonstration purposes — not a replacement for professional inspection.
            </p>

            {/* Primary CTAs */}
            <div className="flex items-center justify-center gap-3 mt-2">
              <Button
                className="bg-blue-600 hover:bg-blue-700 shadow-lg shadow-blue-600/30 hover:scale-[1.03] transition-transform"
                aria-label="Check tire condition"
                onClick={() => onSelectModel('condition')}
              >
                Check Condition
              </Button>
              <Button
                className="bg-emerald-600 hover:bg-emerald-700 shadow-lg shadow-emerald-600/30 hover:scale-[1.03] transition-transform"
                aria-label="Check tire inflation"
                onClick={() => onSelectModel('inflation')}
              >
                Check Inflation
              </Button>
            </div>
          </div>

          {/* Stats Grid (denser + responsive) */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 md:gap-4 mb-2">
            {stats.map((stat, i) => (
              <Card
                key={i}
                className="bg-slate-800/60 border-slate-700 backdrop-blur-sm p-3 md:p-4 transition-all hover:bg-slate-800/80 hover:scale-[1.02] animate-in fade-in duration-700"
                style={{ animationDelay: `${100 + i * 80}ms` }}
              >
                <div className="flex items-center gap-3">
                  <div className="size-9 md:size-10 rounded-xl bg-slate-900/60 grid place-items-center">
                    <stat.icon className={`size-4 md:size-5 ${stat.color}`} />
                  </div>
                  <div>
                    <p className="text-white text-base md:text-lg font-semibold leading-tight">{stat.value}</p>
                    <p className="text-slate-300 text-xs">{stat.label}</p>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        </div>
      </div>

      {/* Model Selection */}
      <section className="max-w-4xl mx-auto px-4 pb-6">
        <div className="text-center space-y-2 mb-4 animate-in slide-in-from-bottom-4 duration-700">
          <h2 className="text-white text-2xl md:text-3xl font-semibold">Choose Your Inspection Type</h2>
          <p className="text-slate-200 max-w-2xl mx-auto">
            Select which aspect of your tire you&apos;d like to analyze using our ML-powered classification models.
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-4 md:gap-5 mb-3">
          {/* Condition Card */}
          <Card
            role="button"
            tabIndex={0}
            onKeyDown={(e) => e.key === 'Enter' && onSelectModel('condition')}
            onClick={() => onSelectModel('condition')}
            className="bg-slate-800/60 border-slate-700 backdrop-blur-sm hover:bg-slate-800/80 hover:border-blue-500/50 hover:shadow-2xl hover:shadow-blue-500/10 transition-all cursor-pointer group animate-in slide-in-from-left duration-700"
          >
            <div className="p-5 md:p-7 space-y-4">
              <div className="size-14 md:size-16 bg-blue-500/10 rounded-2xl grid place-items-center group-hover:bg-blue-500/20 transition-colors animate-in zoom-in-95 duration-500">
                <Eye className="size-7 md:size-8 text-blue-400 group-hover:text-blue-300 transition-colors" />
              </div>
              <div className="space-y-1.5">
                <h3 className="text-white group-hover:text-blue-100 transition-colors">Tire Condition</h3>
                <p className="text-slate-400 group-hover:text-slate-300 transition-colors">
                  Analyze tread wear patterns to determine if your tire is in good condition or needs replacement.
                </p>
              </div>
              <Button
                className="w-full bg-blue-600 hover:bg-blue-700 shadow-lg shadow-blue-600/30 hover:scale-[1.02] transition-transform"
                onClick={(e) => { e.stopPropagation(); onSelectModel('condition'); }}
              >
                Check Condition
              </Button>
            </div>
          </Card>

          {/* Inflation Card */}
          <Card
            role="button"
            tabIndex={0}
            onKeyDown={(e) => e.key === 'Enter' && onSelectModel('inflation')}
            onClick={() => onSelectModel('inflation')}
            className="bg-slate-800/60 border-slate-700 backdrop-blur-sm hover:bg-slate-800/80 hover:border-emerald-500/50 hover:shadow-2xl hover:shadow-emerald-500/10 transition-all cursor-pointer group animate-in slide-in-from-right duration-700"
          >
            <div className="p-5 md:p-7 space-y-4">
              <div className="size-14 md:size-16 bg-emerald-500/10 rounded-2xl grid place-items-center group-hover:bg-emerald-500/20 transition-colors animate-in zoom-in-95 duration-500">
                <Gauge className="size-7 md:size-8 text-emerald-400 group-hover:text-emerald-300 transition-colors" />
              </div>
              <div className="space-y-1.5">
                <h3 className="text-white group-hover:text-emerald-100 transition-colors">Tire Inflation</h3>
                <p className="text-slate-400 group-hover:text-slate-300 transition-colors">
                  Detect whether your tire is properly inflated or requires air pressure adjustment.
                </p>
              </div>
              <Button
                className="w-full bg-emerald-600 hover:bg-emerald-700 shadow-lg shadow-emerald-600/30 hover:scale-[1.02] transition-transform"
                onClick={(e) => { e.stopPropagation(); onSelectModel('inflation'); }}
              >
                Check Inflation
              </Button>
            </div>
          </Card>
        </div>
      </section>

      {/* Footer / About */}
      <footer className="border-t border-white/10">
        <div className="max-w-6xl mx-auto px-4 py-4 grid sm:grid-cols-2 gap-6 animate-in slide-in-from-bottom-4 duration-700">
          <div>
            <h3 className="text-white font-semibold mb-2">About this project</h3>
            <p className="text-slate-300 text-sm leading-relaxed">
              A portfolio demo applying deep learning to tire health: image classification for condition,
              and for inflation checks. Pytorch Training · FastAPI backend · Azure ML serving · React + Tailwind UI.
            </p>
          </div>
          <div className="sm:justify-self-end">
            <div className="flex items-center gap-3">
              <div className="size-9 md:size-10 rounded-full ring-2 ring-white/10 grid place-items-center bg-emerald-600/20 text-emerald-300 font-semibold">
                RY
              </div>
              <div>
                <p className="text-white text-sm font-medium">Rithwik Yenuganti</p>
                <p className="text-slate-400 text-xs">Data Scientist → ML Engineer (PyTorch, Azure)</p>
                <a
                  href="https://github.com/ripp3r-ds/tyre-health-poc"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-xs text-slate-300 hover:text-white underline underline-offset-4"
                >
                  View source on GitHub
                </a>
              </div>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
