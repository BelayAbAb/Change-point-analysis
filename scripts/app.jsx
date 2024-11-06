import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { HeatMap } from 'react-heatmap-grid';

function App() {
    const [oilPrices, setOilPrices] = useState([]);
    const [economicData, setEconomicData] = useState([]);
    const [correlationData, setCorrelationData] = useState({});
    const [events, setEvents] = useState([]);

    useEffect(() => {
        // Fetch data from Flask API
        axios.get('http://localhost:5000/api/oil-prices')
            .then(response => setOilPrices(response.data))
            .catch(error => console.log(error));

        axios.get('http://localhost:5000/api/economic-indicators')
            .then(response => setEconomicData(response.data))
            .catch(error => console.log(error));

        axios.get('http://localhost:5000/api/correlation')
            .then(response => setCorrelationData(response.data))
            .catch(error => console.log(error));

        axios.get('http://localhost:5000/api/events')
            .then(response => setEvents(response.data))
            .catch(error => console.log(error));
    }, []);

    // Prepare oil price data for the chart
    const oilChartData = oilPrices.map(item => ({
        date: item.Date.split('T')[0], // Format date as YYYY-MM-DD
        price: item.Price
    }));

    return (
        <div className="App">
            <h1>Brent Oil Price Analysis Dashboard</h1>

            {/* Oil Price Trend Chart */}
            <div>
                <h2>Brent Oil Price Over Time</h2>
                <ResponsiveContainer width="100%" height={400}>
                    <LineChart data={oilChartData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="date" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Line type="monotone" dataKey="price" stroke="#8884d8" />
                    </LineChart>
                </ResponsiveContainer>
            </div>

            {/* Economic Indicators Correlation */}
            <div>
                <h2>Correlation between Economic Indicators</h2>
                <HeatMap
                    xLabels={Object.keys(correlationData)}
                    yLabels={Object.keys(correlationData)}
                    data={Object.values(correlationData).map(row => Object.values(row))}
                />
            </div>

            {/* Event Timeline */}
            <div>
                <h2>Key Events and Price Impact</h2>
                <ul>
                    {events.map((event, index) => (
                        <li key={index}>
                            <strong>{event.event}</strong> on {event.date} - Price Impact: {event.price_impact}
                        </li>
                    ))}
                </ul>
            </div>
        </div>
    );
}

export default App;
