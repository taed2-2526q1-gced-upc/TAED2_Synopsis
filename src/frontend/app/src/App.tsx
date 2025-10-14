import { useState } from 'react'
import './App.css'

function App() {
  const [selectedProviders, setSelectedProviders] = useState<string[]>(['All providers'])
  const [articleUrl, setArticleUrl] = useState('')
  const [healthStatus, setHealthStatus] = useState<string | null>(null)
  const [isCheckingHealth, setIsCheckingHealth] = useState(false)

  const providers = [
    'All providers',
    'New York Times',
    'BBC',
    'The Guardian',
    'Reuters',
    'Washington Post',
    'CNN',
    'Al Jazeera',
    'Bloomberg'
  ]

  const toggleProvider = (provider: string) => {
    if (provider === 'All providers') {
      setSelectedProviders(['All providers'])
    } else {
      setSelectedProviders(prev => {
        const filtered = prev.filter(p => p !== 'All providers')
        if (filtered.includes(provider)) {
          const newProviders = filtered.filter(p => p !== provider)
          return newProviders.length === 0 ? ['All providers'] : newProviders
        } else {
          return [...filtered, provider]
        }
      })
    }
  }

  const handleSummarize = () => {
    // TODO: Implement summarization logic
    console.log('Summarizing article:', articleUrl, 'with providers:', selectedProviders)
  }

  const handleUploadPdf = () => {
    // TODO: Implement PDF upload logic
    console.log('Uploading PDF')
  }

  const checkHealth = async () => {
    setIsCheckingHealth(true)
    setHealthStatus(null)

    try {
      const response = await fetch('/api/health')
      if (response.ok) {
        const data = await response.json()
        setHealthStatus(`✅ ${JSON.stringify(data, null, 2)}`)
      } else {
        const errorText = await response.text()
        setHealthStatus(`❌ API Error (${response.status}): ${errorText || response.statusText}`)
      }
    } catch (error) {
      setHealthStatus(`❌ Connection Error: ${error instanceof Error ? error.message : 'Unknown error'}`)
    } finally {
      setIsCheckingHealth(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-purple-600 rounded-full flex items-center justify-center">
            <span className="text-white font-bold text-lg">S</span>
          </div>
          <span className="text-white text-xl font-semibold">Synopsis</span>
        </div>

        <nav className="hidden md:flex items-center gap-8">
          <a href="#" className="text-gray-300 hover:text-white transition-colors">How it works</a>
          <a href="#" className="text-gray-300 hover:text-white transition-colors">Providers</a>
          <a href="#" className="text-gray-300 hover:text-white transition-colors">Pricing</a>
          <a href="#" className="text-gray-300 hover:text-white transition-colors">Docs</a>
          <button
            onClick={checkHealth}
            disabled={isCheckingHealth}
            className="bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white px-4 py-2 rounded-lg transition-colors flex items-center gap-2"
          >
            {isCheckingHealth ? (
              <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="m4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
            ) : (
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            )}
            {isCheckingHealth ? 'Checking...' : 'Check Health'}
          </button>
          <button className="bg-purple-600 hover:bg-purple-700 text-white px-4 py-2 rounded-lg transition-colors">
            Sign in
          </button>
        </nav>
      </header>

      {/* Main Content */}
      <main className="flex flex-col items-center justify-center px-6 py-16">
        {/* Hero Section */}
        <div className="text-center mb-16 max-w-4xl">
          <h1 className="text-6xl md:text-7xl font-bold text-white mb-4">
            Synopsis
          </h1>

          <h2 className="text-4xl md:text-4xl font-bold bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent mb-8">
            <span className="text-white">Instant, </span>
            trustworthy news summaries
          </h2>
          <p className="text-xl text-gray-300 max-w-2xl mx-auto">
            Paste a news URL or choose providers, hit Summarize, get a compact,
            neutral summary in seconds.
          </p>
        </div>

        {/* Health Status Display */}
        {healthStatus && (
          <div className="my-6 w-full max-w-2xl">
            <div className={`p-4 rounded-lg border ${healthStatus.startsWith('✅')
                ? 'bg-green-900/20 border-green-500 text-green-300'
                : 'bg-red-900/20 border-red-500 text-red-300'
              }`}>
              <pre className="text-sm whitespace-pre-wrap break-words font-mono">{healthStatus}</pre>
            </div>
          </div>
        )}

        {/* Interactive Form */}
        <div className="w-full max-w-2xl bg-gray-800/50 backdrop-blur-sm rounded-2xl p-8 border border-gray-700">
          {/* Provider Selection */}
          <div className="mb-6">
            <h3 className="text-white text-lg font-medium mb-4">Select providers</h3>
            <div className="flex flex-wrap gap-2">
              {providers.map((provider) => (
                <button
                  key={provider}
                  onClick={() => toggleProvider(provider)}
                  className={`px-4 py-2 rounded-full text-sm font-medium transition-colors ${selectedProviders.includes(provider)
                      ? 'bg-purple-600 text-white'
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                    }`}
                >
                  {provider}
                </button>
              ))}
            </div>
          </div>

          {/* URL Input */}
          <div className="mb-6">
            <h3 className="text-white text-lg font-medium mb-4">Article URL</h3>
            <input
              type="url"
              value={articleUrl}
              onChange={(e) => setArticleUrl(e.target.value)}
              placeholder="Paste article URL here — e.g. https://..."
              className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-purple-500 focus:ring-1 focus:ring-purple-500"
            />
          </div>

          {/* Action Buttons */}
          <div className="flex gap-4">
            <button
              onClick={handleSummarize}
              disabled={!articleUrl.trim()}
              className="flex-1 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white py-3 px-6 rounded-lg font-medium flex items-center justify-center gap-2 transition-colors"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
              Summarize
            </button>
            {/* <button
              onClick={handleUploadPdf}
              className="bg-gray-700 hover:bg-gray-600 text-white py-3 px-6 rounded-lg font-medium flex items-center gap-2 transition-colors"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
              Upload PDF
            </button> */}
          </div>

          {/* Privacy Notice */}
          <p className="text-gray-400 text-sm text-center mt-6">
            We respect your privacy — we fetch only the article content you choose.
          </p>
        </div>


      </main>
    </div>
  )
}

export default App
