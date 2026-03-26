import { useState } from 'react';
import { generateMockFlights } from './utils/mockData';

function App() {
  const [origin, setOrigin] = useState('MOW');
  const [destination, setDestination] = useState('LED');
  const [flights, setFlights] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSearch = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    setFlights([]); // Clear old results

    try {
      // 1. Generate 10 mock flights for the requested route
      const mockFlights = generateMockFlights(origin.toUpperCase(), destination.toUpperCase(), 10);

      // 2. Send them to your local Python XGBoost API
      const response = await fetch('https://flight-recommendation-api.onrender.com/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ranker_id: Math.floor(Math.random() * 1000), // Random session ID
          flights: mockFlights
        })
      });

      if (!response.ok) {
        throw new Error(`API Error: ${response.status}`);
      }

      const data = await response.json();

      // 3. The API returns just the sorted IDs and scores. 
      // We need to map those IDs back to our rich mock flight data to display them.
      const rankedFlights = data.ranked_flights.map(rankedItem => {
        const flightData = mockFlights.find(f => f.Id === rankedItem.Id);
        return { ...flightData, mlScore: rankedItem.score };
      });

      setFlights(rankedFlights);
    } catch (err) {
      console.error(err);
      setError("Failed to fetch recommendations. Is your Python backend running?");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-100 text-slate-800 font-sans">
      {/* Header */}
      <header className="bg-blue-600 text-white p-6 shadow-md">
        <div className="max-w-5xl mx-auto flex items-center justify-between">
          <h1 className="text-2xl font-bold tracking-tight">AI Flight Ranker</h1>
          <span className="bg-blue-800 px-3 py-1 rounded-full text-xs font-semibold tracking-wide">
            Powered by XGBoost
          </span>
        </div>
      </header>

      <main className="max-w-5xl mx-auto p-6 mt-6">
        {/* Search Bar */}
        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 mb-8">
          <form onSubmit={handleSearch} className="flex flex-col md:flex-row gap-4 items-end">
            <div className="flex-1 w-full">
              <label className="block text-sm font-semibold text-slate-600 mb-1">Origin (IATA)</label>
              <input 
                type="text" 
                value={origin} 
                onChange={(e) => setOrigin(e.target.value)}
                className="w-full border border-slate-300 rounded-lg p-3 uppercase focus:ring-2 focus:ring-blue-500 focus:outline-none"
                placeholder="e.g. MOW"
                maxLength={3}
                required
              />
            </div>
            <div className="flex-1 w-full">
              <label className="block text-sm font-semibold text-slate-600 mb-1">Destination (IATA)</label>
              <input 
                type="text" 
                value={destination} 
                onChange={(e) => setDestination(e.target.value)}
                className="w-full border border-slate-300 rounded-lg p-3 uppercase focus:ring-2 focus:ring-blue-500 focus:outline-none"
                placeholder="e.g. LED"
                maxLength={3}
                required
              />
            </div>
            <button 
              type="submit" 
              disabled={isLoading}
              className="w-full md:w-auto bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-8 rounded-lg transition-colors disabled:opacity-50"
            >
              {isLoading ? 'Ranking...' : 'Search Flights'}
            </button>
          </form>
        </div>

        {/* Error State */}
        {error && (
          <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 mb-8 rounded">
            <p>{error}</p>
          </div>
        )}

        {/* Flight Results */}
        <div className="space-y-4">
          {flights.map((flight, index) => (
            <div key={flight.Id} className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 flex flex-col md:flex-row justify-between items-center gap-6 hover:shadow-md transition-shadow">
              
              {/* Left Side: Airline & Route */}
              <div className="flex-1">
                <div className="flex items-center gap-3 mb-2">
                  <span className="bg-slate-100 text-slate-600 font-bold px-3 py-1 rounded-md text-sm border border-slate-200">
                    {flight.legs0_segments0_marketingCarrier_code}
                  </span>
                  {index === 0 && (
                    <span className="bg-green-100 text-green-700 font-bold px-2 py-1 rounded-md text-xs uppercase tracking-wider">
                      Best Choice
                    </span>
                  )}
                </div>
                <p className="text-xl font-bold">{flight.searchRoute.slice(0,3)} → {flight.searchRoute.slice(3,6)}</p>
                <p className="text-slate-500 text-sm mt-1">
                  Class: {flight.legs0_segments0_cabinClass === 2 ? 'Premium' : 'Economy'} • 
                  Duration: {flight.legs0_duration.split('.')[1] || flight.legs0_duration}
                </p>
              </div>

              {/* Right Side: Price & ML Score */}
              <div className="text-right flex flex-row md:flex-col items-center justify-between w-full md:w-auto">
                <div>
                  <p className="text-2xl font-extrabold text-blue-600">
                    ₽{flight.totalPrice.toLocaleString()}
                  </p>
                  <p className="text-xs text-slate-400 mt-1" title="XGBoost raw ranking score">
                    ML Score: {flight.mlScore.toFixed(3)}
                  </p>
                </div>
                <button className="md:mt-4 bg-slate-900 hover:bg-slate-800 text-white px-6 py-2 rounded-lg font-medium transition-colors">
                  Select
                </button>
              </div>

            </div>
          ))}
          
          {flights.length === 0 && !isLoading && !error && (
            <div className="text-center text-slate-500 py-12">
              Enter an origin and destination to see AI-ranked flight recommendations.
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
