import { useState, useEffect } from 'react';

interface EmotionChartProps {
  emotion: string;
  value: number;
  color: string;
}

export default function EmotionChart({ emotion, value, color }: EmotionChartProps) {
  const [displayValue, setDisplayValue] = useState(0);
  const circumference = 2 * Math.PI * 40; // radius = 40
  const strokeDashoffset = circumference * (1 - displayValue);

  useEffect(() => {
    setDisplayValue(0);
    
    const steps = 30; 
    const increment = value / steps;
    const duration = 1000;
    const stepDuration = duration / steps;

    let currentStep = 0;
    const timer = setInterval(() => {
      currentStep++;
      if (currentStep <= steps) {
        setDisplayValue(prev => {
          const next = prev + increment;
          return next > value ? value : next;
        });
      } else {
        clearInterval(timer);
        setDisplayValue(value); 
      }
    }, stepDuration);

    return () => clearInterval(timer);
  }, [value]);

  return (
    <div className="flex flex-col items-center">
      <div className="relative w-24 h-24">
        <svg className="w-full h-full transform -rotate-90">
          <circle
            cx="48"
            cy="48"
            r="40"
            stroke="currentColor"
            strokeWidth="8"
            fill="transparent"
            className="text-gray-700"
          />
          <circle
            cx="48"
            cy="48"
            r="40"
            stroke="currentColor"
            strokeWidth="8"
            fill="transparent"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            className={color}
            style={{ transition: 'stroke-dashoffset 0.5s ease' }}
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-white text-lg font-semibold transition-all duration-75">
            {Math.round(displayValue * 100)}%
          </span>
        </div>
      </div>
      <span className="mt-2 text-gray-300 capitalize text-sm">
        {emotion}
      </span>
    </div>
  );
}