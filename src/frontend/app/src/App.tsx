import { useState } from 'react'
import './App.css'
import logo from '/src/assets/logo.png'
import logoWhite from '/src/assets/logo-white.png'
import EmotionChart from './components/EmotionChart'
import newyorktimesLogo from '/src/assets/The_New_York_Times_icon.webp'
import reutersLogo from '/src/assets/Reuters_2024_symbol.png'
import theguardianLogo from '/src/assets/the-guardian.png'
import bbcLogo from '/src/assets/bbc-logo.png'
import thewpLogo from '/src/assets/the-wp.png'
import apLogo from '/src/assets/ap-logo.png'
import bloombergLogo from '/src/assets/bloomberg.svg'
import ftLogo from '/src/assets/ft.png'

const providerLogos: Record<string, string> = {
  "The New York Times": newyorktimesLogo,
  "Reuters": reutersLogo,
  "The Guardian": theguardianLogo,
  "BBC News": bbcLogo,
  "Washington Post": thewpLogo,
  "Associated Press": apLogo,
  "Bloomberg": bloombergLogo,
  "Financial Times": ftLogo,
}


function App() {
  // State variables
  const [articleUrl, setArticleUrl] = useState('')
  const [summary, setSummary] = useState<string | null>(null)
  const [fullArticle, setFullArticle] = useState<string | null>(null)
  const [showFullArticle, setShowFullArticle] = useState(false)
  const [title, setTitle] = useState<string | null>(null)
  const [healthStatus, setHealthStatus] = useState<string | null>(null)
  const [isCheckingHealth, setIsCheckingHealth] = useState(false)
  const [isSummarizing, setIsSummarizing] = useState(false)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [sentimentAnalysis, setSentimentAnalysis] = useState<Record<string, number> | null>(null)
  const [urlError, setUrlError] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Helpers
  const isValidUrl = (url: string): boolean => {
    try {
      const parsedUrl = new URL(url);
      return parsedUrl.protocol === 'http:' || parsedUrl.protocol === 'https:';
    } catch (e) {
      return false;
    }
  }

  const handleReset = () => {
    setArticleUrl('');
    setSummary(null);
    setFullArticle(null);
    setShowFullArticle(false);
    setTitle(null);
    setSentimentAnalysis(null);
    setError(null);
  }

  // Main API Calls
  const checkHealth = async () => {
    setIsCheckingHealth(true)
    setHealthStatus(null)
    try {
      const response = await fetch('/api/health/')
      if (response.ok) {
        setHealthStatus('✅ Online')
      } else {
        setHealthStatus('❌ Offline')
      }
    } catch (error) {
      setHealthStatus('❌ Offline')
    } finally {
      setIsCheckingHealth(false)
    }
  }

  const handleSummarize = async () => {
    setUrlError(null);
    setError(null);
    if (!isValidUrl(articleUrl)) {
      setUrlError('Please enter a valid link!');
      return;
    }
    setIsSummarizing(true)
    try {
      const response = await fetch('/api/summarize/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          url: articleUrl,
        })
      });

      if (response.ok) {
        const data = await response.json();
        setSummary(data.summary);
        setFullArticle(data.full_article);
        setTitle(data.title);
      } else {
        const errorData = await response.json();
        setError(errorData.detail || 'An error happened. Try with a different article or again in a few minutes.');
        setSummary(null);
        setFullArticle(null);
        setTitle(null);
      }
    } catch (err) {
      setError('Error: Could not connect to the server');
      setSummary(null);
      setTitle(null);
    } finally {
      setIsSummarizing(false);
    }
  }

  const handleAnalyze = async () => {
    setIsAnalyzing(true);
    try {
      const response = await fetch('/api/analyze/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          text: summary
        })
      });

      if (response.ok) {
        const data = await response.json();
        setSentimentAnalysis(data.probabilities);
      } else {
        const errorData = await response.json();
        setError(errorData.detail || 'An error happened while analyzing sentiment. Please try again.');
      }
    } catch (err) {
      setError('Error: Could not connect to the server');
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900">
      {/* Navbar */}
      <header className="flex items-center justify-between px-8 py-6">
        <div className="flex items-center gap-3">
          <img src={logo} alt="Synopsis Logo" className="h-14" />
          <span className="text-white text-2xl font-semibold">Synopsis</span>
        </div>
        <nav className="hidden md:flex items-center gap-8">
          <a href="#how" className="text-gray-300 hover:text-white transition-colors">How it works</a>
          <a href="#providers" className="text-gray-300 hover:text-white transition-colors">Providers</a>
          <a href="#pricing" className="text-gray-300 hover:text-white transition-colors">Pricing</a>
          <button
            onClick={checkHealth}
            disabled={isCheckingHealth}
            className={`
              ${isCheckingHealth
                ? 'bg-gray-600'
                : healthStatus === '✅ Online'
                  ? 'bg-green-600 hover:bg-green-700'
                  : healthStatus === '❌ Offline'
                    ? 'bg-red-600 hover:bg-red-700'
                    : 'bg-green-600 hover:bg-green-700'}
              disabled:bg-gray-600 disabled:cursor-not-allowed text-white px-4 py-2 rounded-lg transition-colors flex items-center gap-2`
            }
          >
            {isCheckingHealth ? (
              <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="m4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
            ) : healthStatus === '✅ Online' ? (
              <svg className="w-4 h-4 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            ) : healthStatus === '❌ Offline' ? (
              <svg className="w-4 h-4 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            ) : (
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            )}
            {isCheckingHealth
              ? 'Checking...'
              : healthStatus === '✅ Online'
                ? 'Online'
                : healthStatus === '❌ Offline'
                  ? 'Offline'
                  : 'Check Status'}
          </button>
        </nav>
      </header>

      <main className="flex flex-col items-center justify-center px-6 py-16">
        {/* Title */}
        <div className="text-center mb-16 max-w-4xl">
          <h1 className="text-6xl md:text-7xl font-bold text-white mb-4">
            Synopsis
          </h1>

          <h2 className="text-4xl md:text-4xl font-bold bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent mb-8">
            <span className="text-white">Instant, </span>
            trustworthy news summaries
          </h2>
          <p className="text-xl text-gray-300 max-w-2xl mx-auto">
            Paste a news URL, hit Summarize, get a compact, neutral, and informative
            summary in seconds.
          </p>
        </div>

        {/* Main Form */}
        <div className="w-full max-w-2xl bg-gray-800/50 backdrop-blur-sm rounded-2xl p-8 border border-gray-700 mt-14">
          {!summary || error ? (
            <>
              <div className="mb-6">
                <input
                  type="url"
                  value={articleUrl}
                  onChange={(e) => {
                    setArticleUrl(e.target.value);
                    setUrlError(null);
                    setError(null);
                  }}
                  placeholder="Paste article URL here — e.g. https://..."
                  className={`w-full px-4 py-3 bg-gray-700 border rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-1 transition-colors ${urlError
                    ? 'border-red-500 focus:border-red-500 focus:ring-red-500'
                    : 'border-gray-600 focus:border-purple-500 focus:ring-purple-500'
                    }`}
                  disabled={isSummarizing}
                />
                {urlError && (
                  <p className="mt-4 text-red-400 text-base font-semibold flex items-center gap-2">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    {urlError}
                  </p>
                )}
                {error && (
                  <p className="mt-4 text-red-400 text-base font-semibold flex items-center gap-2">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    {error}
                  </p>
                )}
              </div>
              <div className="flex gap-4">
                <button
                  onClick={handleSummarize}
                  disabled={!articleUrl.trim() || isSummarizing}
                  className="flex-1 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white py-3 px-6 rounded-lg font-medium flex items-center justify-center gap-2 transition-colors"
                >
                  {isSummarizing ? (
                    <>
                      <svg className="animate-spin h-5 w-5" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      <span>Summarizing...</span>
                    </>
                  ) : (
                    <>
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                      </svg>
                      <span>Summarize</span>
                    </>
                  )}
                </button>
              </div>
            </>
          ) : (
            <>
              <div className="prose prose-invert max-w-none mb-6">
                <h3 className="text-2xl md:text-4xl font-serif font-bold mb-6 bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent">{title}</h3>
                <div className="text-gray-300 text-justify">
                  {(showFullArticle ? fullArticle : summary)?.split('\n').map((paragraph, index) => (
                    <p key={index} className={`${showFullArticle ? 'mb-6' : 'mb-2'}`}>
                      {paragraph}
                    </p>
                  ))}
                </div>
              </div>
              <div className="flex flex-col gap-4 mt-4">
                {!sentimentAnalysis ? (
                  <button
                    onClick={handleAnalyze}
                    disabled={isAnalyzing}
                    className="flex-1 bg-gray-800 hover:bg-purple-700 text-purple-300 py-3 px-6 rounded-lg font-medium flex items-center justify-center gap-2 transition-colors border border-purple-600 disabled:bg-gray-700 disabled:text-gray-500 disabled:border-gray-600 disabled:cursor-not-allowed"
                  >
                    {isAnalyzing ? (
                      <>
                        <svg className="animate-spin h-5 w-5" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        <span>Analyzing...</span>
                      </>
                    ) : (
                      <>
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.828 14.828a4 4 0 01-5.656 0M9 10h.01M15 10h.01M12 2a10 10 0 100 20 10 10 0 000-20z" />
                        </svg>
                        Check Neutrality
                      </>
                    )}
                  </button>
                ) : (
                  <div className="flex-1 bg-gray-800/50 rounded-lg p-6 border border-purple-600">
                    <h4 className="text-white font-medium mb-4 text-center">Neutrality Analysis</h4>
                    <div className="grid grid-cols-3 gap-4">
                      {Object.entries(sentimentAnalysis)
                        .sort(([, a], [, b]) => b - a)
                        .slice(0, 3)
                        .map(([emotion, probability], index) => (
                          <EmotionChart
                            key={emotion}
                            emotion={emotion}
                            value={probability}
                            color={
                              index === 0
                                ? "text-purple-400"
                                : index === 1
                                  ? "text-blue-400"
                                  : "text-indigo-400"
                            }
                          />
                        ))
                      }
                    </div>
                  </div>
                )}
                <button
                  onClick={() => setShowFullArticle(!showFullArticle)}
                  className="flex-1 bg-purple-600 hover:bg-purple-700 text-white py-3 px-6 rounded-lg font-medium flex items-center justify-center gap-2 transition-colors"
                >
                  {showFullArticle ? (
                    <>
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
                      </svg>
                      Show Summary
                    </>
                  ) : (
                    <>
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                      </svg>
                      Show Full Article
                    </>
                  )}
                </button>
                <button
                  onClick={handleReset}
                  className="flex-1 bg-gray-700 hover:bg-gray-600 text-white py-3 px-6 rounded-lg font-medium flex items-center justify-center gap-2 transition-colors"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                  Summarize Another Article
                </button>

              </div>
            </>
          )}
        </div>

        {/* How it Works Section */}
        <div id="how" className="mt-40 w-full max-w-4xl">
          <div className="text-center mb-12">
            <h2 className="text-2xl md:text-3xl font-bold bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent mb-4">
              How It Works
            </h2>
            <p className="text-lg text-gray-300 max-w-2xl mx-auto">
              Three simple steps to get your article summary
            </p>
          </div>
          <div className="relative flex justify-between items-center gap-4">
            <div className="flex-1 bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700 hover:border-purple-500 transition-colors z-10">
              <div className="flex flex-col items-center text-center gap-4">
                <div className="w-16 h-16 bg-gray-700 rounded-lg flex items-center justify-center group-hover:bg-purple-900/50 transition-colors">
                  <svg className="w-8 h-8 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                </div>
                <h3 className="text-lg font-semibold text-white">Input</h3>
                <p className="text-gray-300 text-sm">
                  Paste any news article URL from your favorite provider
                </p>
              </div>
            </div>
            <div className="flex-none w-12 h-px bg-gradient-to-r from-purple-500 to-blue-500 relative">
              <div className="absolute -right-1 -top-1 w-2 h-2 rotate-45 border-t border-r border-blue-500"></div>
            </div>
            <div className="flex-1 bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700 hover:border-purple-500 transition-colors z-10">
              <div className="flex flex-col items-center text-center gap-4">
                <div className="w-16 h-16 bg-gray-700 rounded-lg flex items-center justify-center group-hover:bg-purple-900/50 transition-colors">
                  <svg className="w-8 h-8 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                  </svg>
                </div>
                <h3 className="text-lg font-semibold text-white">Scrape</h3>
                <p className="text-gray-300 text-sm">
                  Our API first scrapes the URL to extract the article's content
                </p>
              </div>
            </div>
            <div className="flex-none w-12 h-px bg-gradient-to-r from-purple-500 to-blue-500 relative">
              <div className="absolute -right-1 -top-1 w-2 h-2 rotate-45 border-t border-r border-blue-500"></div>
            </div>
            <div className="flex-1 bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700 hover:border-purple-500 transition-colors z-10">
              <div className="flex flex-col items-center text-center gap-4">
                <div className="w-16 h-16 bg-gray-700 rounded-lg flex items-center justify-center group-hover:bg-purple-900/50 transition-colors">
                  <svg className="w-8 h-8 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                  </svg>
                </div>
                <h3 className="text-lg font-semibold text-white">Model</h3>
                <p className="text-gray-300 text-sm">
                  Our fine-tuned BART model generates a concise and accurate summary
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Providers Section */}
        <div id="providers" className="mt-40 w-full max-w-4xl">
          <div className="text-center mb-12">
            <h2 className="text-2xl md:text-3xl font-bold bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent mb-4">Universal News Coverage</h2>
            <p className="text-lg text-gray-300 max-w-2xl mx-auto">
              Our advanced content extraction algorithm can parse and summarize articles from any news provider
            </p>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
            {Object.entries(providerLogos).map(([provider, logoSrc]) => (
              <div key={provider} className="aspect-video bg-gray-800/50 backdrop-blur-sm rounded-xl p-4 flex items-center justify-center border border-gray-700 hover:border-purple-500 transition-colors group">
                <div className="w-full flex flex-col items-center gap-3">
                  <img
                    src={logoSrc}
                    alt={provider}
                    className="h-8 w-auto object-contain opacity-75 group-hover:opacity-100 transition-opacity filter invert"
                  />
                  <span className="text-sm text-center text-gray-400 group-hover:text-purple-400 transition-colors">{provider}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Pricing Section */}
        <div id="pricing" className="mt-36 w-full max-w-md">
          <div className="text-center mb-8">
            <h2 className="text-2xl md:text-3xl font-bold bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent mb-4">Simple Pricing</h2>
            <p className="text-gray-300">No hidden fees. No commitment.</p>
          </div>
          <div className="bg-gray-800/50 backdrop-blur-sm rounded-2xl p-8 border border-gray-700">
            <div className="text-center">
              <h3 className="text-xl font-semibold text-white mb-2">Free Plan</h3>
              <div className="flex justify-center items-baseline mb-4">
                <span className="text-5xl font-extrabold text-white">0€</span>
                <span className="text-gray-400 ml-1">/forever</span>
              </div>
              <p className="text-gray-300 mb-6">We cover all the costs of the queries</p>
              <ul className="text-left space-y-4 mb-8">
                <li className="flex items-center text-gray-300">
                  <svg className="w-5 h-5 text-green-400 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  Unlimited News Summarization
                </li>
                <li className="flex items-center text-gray-300">
                  <svg className="w-5 h-5 text-green-400 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  High-Quality AI Summaries
                </li>
                <li className="flex items-center text-gray-300">
                  <svg className="w-5 h-5 text-green-400 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  No API Key Required
                </li>
              </ul>
              <a href="#" className="w-full bg-purple-600 hover:bg-purple-700 text-white py-3 px-6 rounded-lg font-medium transition-colors">
                Start Using Now
              </a>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-gray-700 mt-16 bg-gray-900">
        <div className="max-w-7xl mx-auto px-4 py-8">
          <div className="flex flex-col items-center gap-4">
            <img src={logoWhite} alt="Synopsis Logo" className="h-12" />
            <div className="mt-4 pt-4 border-t border-gray-800 w-full">
              <p className="text-center text-gray-400 text-sm">
                © {new Date().getFullYear()} Synopsis. All rights reserved.
              </p>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default App
