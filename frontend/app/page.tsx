'use client';

import React, { useState } from 'react';
import axios from 'axios';

interface KeyIndicator {
  token: string;
  importance: number;
}

interface Explanation {
  reasons: string[];
  key_indicators: KeyIndicator[];
  shap?: {
    status: string;
    execution_time_ms: number;
    top_features: Array<{ word: string; importance: number; direction: string }>;
  };
  lime?: {
    status: string;
    execution_time_ms: number;
    top_features: Array<{ word: string; weight: number; direction: string }>;
  };
}

interface PredictionResult {
  prediction: string;
  classification_label: number;
  phishing_probability: number;
  legitimate_probability: number;
  confidence_score: number;
  explanation: Explanation;
}

export default function Home() {
  const [sender, setSender] = useState('security-update@paypal-support.com');
  const [recipient, setRecipient] = useState('user@target.com');
  const [subject, setSubject] = useState('URGENT: Your account has been suspended! Verify now.');
  const [body, setBody] = useState('Dear customer, unauthorized login detected. Click here immediately to restore your account access or it will be permanently closed.');
  
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState('');

  const handleScan = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setResult(null);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';
      const response = await axios.post(`${apiUrl}/predict`, {
        from: sender,
        to: recipient,
        subject: subject,
        body: body,
      });
      setResult(response.data);
    } catch (err: unknown) {
      console.error(err);
      let backendError = 'Unknown error occurred';

      if (axios.isAxiosError<{ detail?: string }>(err)) {
        backendError = err.response?.data?.detail || err.message;
      } else if (err instanceof Error) {
        backendError = err.message;
      }

      setError(`Backend Error: ${backendError}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100 flex flex-col items-center py-10 px-4">
      {/* Header */}
      <div className="max-w-4xl w-full text-center mb-10">
        <span className="px-3 py-1 text-xs font-semibold bg-blue-500/10 text-blue-400 rounded-full border border-blue-500/20">
          MSc Research Project • Explainable DistilBERT AI Engine
        </span>
        <h1 className="text-4xl font-extrabold tracking-tight mt-3 text-white">
          AI Email Phishing Detection System
        </h1>
        <p className="text-slate-400 mt-2 text-sm max-w-xl mx-auto">
          Analyze suspicious emails in real-time with transformer-based attention explainability.
        </p>
      </div>

      {/* Main Container */}
      <div className="max-w-4xl w-full grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* Input Form */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl h-fit">
          <h2 className="text-lg font-semibold mb-4 text-slate-200">Email Payload</h2>
          <form onSubmit={handleScan} className="space-y-4">
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1">Sender Address</label>
              <input
                type="text"
                value={sender}
                onChange={(e) => setSender(e.target.value)}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-blue-500"
                required
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1">Recipient Address</label>
              <input
                type="text"
                value={recipient}
                onChange={(e) => setRecipient(e.target.value)}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-blue-500"
                required
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1">Subject Line</label>
              <input
                type="text"
                value={subject}
                onChange={(e) => setSubject(e.target.value)}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-blue-500"
                required
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1">Email Body</label>
              <textarea
                rows={4}
                value={body}
                onChange={(e) => setBody(e.target.value)}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-blue-500 resize-none"
                required
              />
            </div>
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-blue-600 hover:bg-blue-500 text-white font-medium py-2.5 rounded-lg text-sm transition-colors duration-200 shadow-lg shadow-blue-600/20 disabled:opacity-50"
            >
              {loading ? 'Analyzing with DistilBERT...' : 'Scan & Explain Email'}
            </button>
          </form>
        </div>

        {/* Results Panel */}
        <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl flex flex-col justify-between">
          <div>
            <h2 className="text-lg font-semibold mb-4 text-slate-200">Analysis & Explainability</h2>

            {error && (
              <div className="bg-red-500/10 border border-red-500/20 text-red-400 p-4 rounded-xl text-sm mb-4 break-words">
                {error}
              </div>
            )}

            {!result && !error && !loading && (
              <div className="h-64 flex flex-col items-center justify-center text-center text-slate-500 border border-dashed border-slate-800 rounded-xl p-6">
                <p className="text-sm">Enter an email payload and click <strong>Scan & Explain Email</strong> to inspect model decisions.</p>
              </div>
            )}

            {loading && (
              <div className="h-64 flex flex-col items-center justify-center text-center text-slate-400">
                <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-500 mb-3"></div>
                <p className="text-sm">Extracting transformer attention weights...</p>
              </div>
            )}

            {result && !loading && (
              <div className="space-y-5">
                {/* Status Badge */}
                <div className={`p-4 rounded-xl border flex items-center justify-between ${
                  result.prediction === 'Phishing' 
                    ? 'bg-red-500/10 border-red-500/30 text-red-400' 
                    : 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
                }`}>
                  <div>
                    <span className="text-xs uppercase tracking-wider font-semibold opacity-80">Verdict</span>
                    <h3 className="text-2xl font-bold">{result.prediction}</h3>
                  </div>
                  <div className="text-right">
                    <span className="text-xs uppercase tracking-wider font-semibold opacity-80">Confidence</span>
                    <p className="text-xl font-bold">{result.confidence_score}%</p>
                  </div>
                </div>

                {/* Probability Breakdown */}
                <div className="space-y-2 bg-slate-950 p-3.5 rounded-xl border border-slate-800">
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-red-400 font-medium">Phishing Risk</span>
                    <span>{result.phishing_probability}%</span>
                  </div>
                  <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                    <div className="bg-red-500 h-full transition-all duration-500" style={{ width: `${result.phishing_probability}%` }}></div>
                  </div>
                </div>

                {/* Why this decision? (Explainability Box) */}
                <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-3">
                  <h4 className="text-xs font-semibold uppercase tracking-wider text-blue-400 flex items-center gap-2">
                    <span>🧠</span> Model Explainability & Reasoning
                  </h4>
                  <ul className="space-y-2 text-xs text-slate-300">
                    {result.explanation.reasons.map((reason, index) => (
                      <li key={index} className="flex items-start gap-2">
                        <span className="text-blue-500 mt-0.5">•</span>
                        <span>{reason}</span>
                      </li>
                    ))}
                  </ul>

                  {/* Key Token Triggers */}
                  {result.explanation.key_indicators && result.explanation.key_indicators.length > 0 && (
                    <div className="mt-3 pt-3 border-t border-slate-800">
                      <span className="text-[11px] font-medium text-slate-400 block mb-2">Key Attention Triggers:</span>
                      <div className="flex flex-wrap gap-1.5">
                        {result.explanation.key_indicators.map((ind, idx) => (
                          <span key={idx} className="bg-slate-800 text-slate-200 px-2 py-1 rounded text-[11px] border border-slate-700">
                            {ind.token} <span className="text-blue-400 font-mono ml-1">({ind.importance}%)</span>
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {result.explanation.shap && (
                    <div className="mt-3 pt-3 border-t border-slate-800">
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-[11px] font-medium text-slate-400">SHAP Features</span>
                        <span className="text-[10px] text-slate-500">
                          {result.explanation.shap.status} • {result.explanation.shap.execution_time_ms} ms
                        </span>
                      </div>
                      {result.explanation.shap.status === 'success' ? (
                        <div className="flex flex-wrap gap-1.5">
                          {result.explanation.shap.top_features.slice(0, 8).map((feature, idx) => (
                            <span key={idx} className="bg-slate-800 text-slate-200 px-2 py-1 rounded text-[11px] border border-slate-700" title={feature.direction}>
                              {feature.word} <span className="text-violet-400 font-mono ml-1">({feature.importance.toFixed(4)})</span>
                            </span>
                          ))}
                        </div>
                      ) : (
                        <p className="text-[11px] text-red-400">SHAP explanation unavailable.</p>
                      )}
                    </div>
                  )}

                  {result.explanation.lime && (
                    <div className="mt-3 pt-3 border-t border-slate-800">
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-[11px] font-medium text-slate-400">LIME Features</span>
                        <span className="text-[10px] text-slate-500">
                          {result.explanation.lime.status} • {result.explanation.lime.execution_time_ms} ms
                        </span>
                      </div>
                      {result.explanation.lime.status === 'success' ? (
                        <div className="flex flex-wrap gap-1.5">
                          {result.explanation.lime.top_features.slice(0, 8).map((feature, idx) => (
                            <span key={idx} className="bg-slate-800 text-slate-200 px-2 py-1 rounded text-[11px] border border-slate-700" title={feature.direction}>
                              {feature.word} <span className="text-amber-400 font-mono ml-1">({feature.weight.toFixed(4)})</span>
                            </span>
                          ))}
                        </div>
                      ) : (
                        <p className="text-[11px] text-red-400">LIME explanation unavailable.</p>
                      )}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          <div className="text-xs text-slate-500 text-center mt-6 pt-4 border-t border-slate-800">
            MSc Dissertation Project • FastAPI & Next.js Integration
          </div>
        </div>
      </div>
    </main>
  );
}