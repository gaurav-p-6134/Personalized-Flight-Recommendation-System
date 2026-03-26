// src/utils/mockData.js

const AIRLINES = ["SU", "S7", "U6", "DP", "UT"];

function generateRandomDuration() {
  const hours = Math.floor(Math.random() * 8) + 2; // 2 to 9 hours
  const minutes = Math.floor(Math.random() * 60);
  // Format: "0.HH:MM:00"
  return `0.${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:00`;
}

export function generateMockFlights(origin, destination, count = 10) {
  const flights = [];
  const route = `${origin}${destination}`;

  for (let i = 0; i < count; i++) {
    const isPremium = Math.random() > 0.8;
    const basePrice = Math.floor(Math.random() * 40000) + 10000; // 10k to 50k
    const price = isPremium ? basePrice * 2.5 : basePrice;
    const carrier = AIRLINES[Math.floor(Math.random() * AIRLINES.length)];

    flights.push({
      Id: `flight_${origin}_${destination}_${Math.random().toString(36).substr(2, 9)}`,
      totalPrice: Math.floor(price),
      taxes: Math.floor(price * 0.15),
      legs0_duration: generateRandomDuration(),
      legs0_segments0_marketingCarrier_code: carrier,
      legs0_segments0_cabinClass: isPremium ? 2 : 3,
      searchRoute: route,
      // Adding a few required numeric defaults so XGBoost doesn't complain
      isVip: isPremium ? 1 : 0,
      legs0_segments0_baggageAllowance_quantity: Math.floor(Math.random() * 2),
      legs1_segments0_baggageAllowance_quantity: 0,
      miniRules0_monetaryAmount: 0,
      miniRules1_monetaryAmount: 0,
    });
  }

  return flights;
}